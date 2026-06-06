"""Evaluate DMPSP routes across a benchmark set of molecules.

Reports route quality metrics:
- Structural: SMILES validity, step count, termination
- Chemical: reaction class range, condition validity (temperature/pressure/time/ratio)
- Model: objective scores, planning time, action diversity
- Summary table across all molecules

Usage:
    python scripts/evaluate.py \
        --checkpoint_dir checkpoints/ \
        --device cuda \
        [--smiles "SMILES1" "SMILES2" ...] \
        [--smiles_file path/to/smiles.txt] \
        [--weights_json '{"yield":0.4,"cost":0.3,"safety":0.3}'] \
        [--max_steps 5] \
        [--out_json results/evaluation.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpsp.encoder import build_encoder
from dmpsp.planner import DMPSPPlanner, PlannerConfig, load_models
from dmpsp.utils import setup_logging, validate_smiles

logger = logging.getLogger(__name__)

# Pharma benchmark molecules (name → SMILES)
BENCHMARK_MOLECULES: dict[str, str] = {
    "aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol":  "CC(=O)Nc1ccc(O)cc1",
    "caffeine":     "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "lidocaine":    "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    "metformin":    "CN(C)C(=N)NC(=N)N",
    "atenolol":     "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
}

# USPTO reaction class names (10 main classes)
_RXNCLASS_NAMES = {
    0: "C-C bond formation",
    1: "Heteroatom alkylation",
    2: "Acylation",
    3: "C-N coupling",
    4: "C-O coupling",
    5: "C-S coupling",
    6: "Functional group interconversion",
    7: "Functional group addition",
    8: "Reduction",
    9: "Oxidation",
}
_N_USPTO_CLASSES = 10

# Physical condition ranges (post-training, after proper normalization)
_COND_EXPECTED_RANGE = (-3.0, 3.0)


@dataclass
class RouteMetrics:
    """Quality metrics for a single planned route."""
    name: str
    smiles: str
    n_steps: int
    terminated_at_target: bool
    planning_time_s: float
    objective_scores: dict[str, float]
    condition_validity: dict[str, bool]
    reaction_class_ids: list[int]
    reaction_class_names: list[str]
    action_diversity: float
    warnings: list[str]


def _check_condition_validity(steps: list[dict]) -> tuple[dict[str, bool], list[str]]:
    """Check if action conditions are within expected physical ranges."""
    warnings: list[str] = []
    lo, hi = _COND_EXPECTED_RANGE

    fields = ["temperature_norm", "pressure_norm", "time_norm", "reagent_ratio"]
    validity: dict[str, bool] = {f: True for f in fields}

    for step in steps:
        for f in fields:
            v = step.get(f, 0.0)
            if not (lo <= v <= hi):
                validity[f] = False

    for f, ok in validity.items():
        if not ok:
            warnings.append(
                f"{f} out of expected [{lo}, {hi}] — model may need more training"
            )

    return validity, warnings


def _action_diversity(steps: list[dict]) -> float:
    """Std-dev across continuous action fields as a diversity proxy."""
    if not steps:
        return 0.0
    vals: list[float] = []
    for step in steps:
        for f in ["temperature_norm", "pressure_norm", "time_norm", "reagent_ratio"]:
            vals.append(step.get(f, 0.0))
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return var ** 0.5


def evaluate_molecule(
    name: str,
    smiles: str,
    planner: DMPSPPlanner,
    objective_weights: dict[str, float],
    max_steps: int,
) -> RouteMetrics:
    """Run planner on one molecule and compute quality metrics."""
    if not validate_smiles(smiles):
        return RouteMetrics(
            name=name, smiles=smiles, n_steps=0, terminated_at_target=False,
            planning_time_s=0.0, objective_scores={}, condition_validity={},
            reaction_class_ids=[], reaction_class_names=[],
            action_diversity=0.0,
            warnings=[f"Invalid SMILES: {smiles!r}"],
        )

    t0 = time.perf_counter()
    route = planner.plan(smiles, objective_weights, max_steps=max_steps)
    elapsed = time.perf_counter() - t0

    # Build step dicts for analysis
    steps_out = [
        {
            "step": i + 1,
            "current_smiles": state.current_smiles,
            "reaction_class_id": max(0, int(round(action.temperature_norm * 0 + action.reaction_class_id))),
            "temperature_norm": action.temperature_norm,
            "pressure_norm": action.pressure_norm,
            "time_norm": action.time_norm,
            "reagent_ratio": action.reagent_ratio,
        }
        for i, (state, action) in enumerate(route.steps)
    ]

    condition_validity, warnings = _check_condition_validity(steps_out)

    # Check reaction class IDs
    rxn_ids = [
        max(0, int(round(action.reaction_class_id)))
        for _, action in route.steps
    ]
    rxn_names = [_RXNCLASS_NAMES.get(rid % _N_USPTO_CLASSES, f"class_{rid}") for rid in rxn_ids]

    for rid in rxn_ids:
        if rid >= 100:
            warnings.append(f"reaction_class_id={rid} exceeds USPTO range — untrained model")

    # Check objective scores in [0, 1]
    for obj, score in route.objective_scores.items():
        if not (0.0 <= score <= 1.0):
            warnings.append(f"objective {obj}={score:.4f} outside [0,1]")

    terminated = (
        route.terminal_smiles == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        if Chem.MolFromSmiles(smiles) else False
    )

    return RouteMetrics(
        name=name,
        smiles=smiles,
        n_steps=route.n_steps,
        terminated_at_target=terminated,
        planning_time_s=round(elapsed, 3),
        objective_scores={k: round(v, 4) for k, v in route.objective_scores.items()},
        condition_validity=condition_validity,
        reaction_class_ids=rxn_ids,
        reaction_class_names=rxn_names,
        action_diversity=round(_action_diversity(steps_out), 4),
        warnings=warnings,
    )


def print_summary_table(results: list[RouteMetrics]) -> None:
    """Print a formatted comparison table to stdout."""
    header = f"{'Molecule':<16} {'Steps':>5} {'Yield':>6} {'Cost':>6} {'Safety':>6} {'Manuf':>6} {'Time(s)':>8} {'Warnings':>4}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        yield_s = r.objective_scores.get("yield", 0.0)
        cost_s = r.objective_scores.get("cost", 0.0)
        safety_s = r.objective_scores.get("safety", 0.0)
        manuf_s = r.objective_scores.get("manufacturability", 0.0)
        n_warn = len(r.warnings)
        print(
            f"{r.name:<16} {r.n_steps:>5} {yield_s:>6.3f} {cost_s:>6.3f} "
            f"{safety_s:>6.3f} {manuf_s:>6.3f} {r.planning_time_s:>8.2f} {n_warn:>4}"
        )
    print("=" * len(header))

    # Aggregate
    all_warnings = sum(len(r.warnings) for r in results)
    avg_time = sum(r.planning_time_s for r in results) / max(len(results), 1)
    print(f"\nTotal molecules: {len(results)}")
    print(f"Avg planning time: {avg_time:.2f}s")
    print(f"Total warnings: {all_warnings}")
    if all_warnings > 0:
        print("\nNote: condition warnings expected with <100K training steps.")
        print("Re-run after full training on Kaggle/vast.ai.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DMPSP routes across benchmark molecules."
    )
    parser.add_argument(
        "--smiles", nargs="+", default=None,
        help="One or more SMILES strings to evaluate. Defaults to built-in pharma benchmark.",
    )
    parser.add_argument(
        "--smiles_file", type=Path, default=None,
        help="Text file with one SMILES per line (optionally 'name SMILES' per line).",
    )
    parser.add_argument(
        "--checkpoint_dir", type=Path, required=True,
        help="Directory with trained model checkpoints.",
    )
    parser.add_argument(
        "--model_config", type=Path, default=Path(__file__).parent.parent / "configs" / "model.yaml",
    )
    parser.add_argument(
        "--device", default="cpu",
    )
    parser.add_argument(
        "--max_steps", type=int, default=5,
        help="Max synthesis steps per molecule. Default: 5.",
    )
    parser.add_argument(
        "--n_candidates", type=int, default=64,
    )
    parser.add_argument(
        "--weights_json",
        default='{"yield":0.3,"cost":0.2,"safety":0.2,"manufacturability":0.15,"fto_risk":0.15}',
        help="JSON objective weights.",
    )
    parser.add_argument(
        "--out_json", type=Path, default=None,
        help="Write full results to this JSON file.",
    )
    parser.add_argument(
        "--log_level", default="WARNING",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Build molecule set
    molecules: dict[str, str] = {}
    if args.smiles_file and args.smiles_file.exists():
        for line in args.smiles_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                molecules[parts[0]] = parts[1]
            else:
                molecules[parts[0]] = parts[0]
    if args.smiles:
        for i, s in enumerate(args.smiles):
            molecules[f"mol_{i+1}"] = s
    if not molecules:
        molecules = BENCHMARK_MOLECULES

    # Parse weights
    try:
        objective_weights: dict[str, float] = json.loads(args.weights_json)
    except json.JSONDecodeError as exc:
        logger.error("Invalid --weights_json: %s", exc)
        sys.exit(1)

    # Load models
    if not args.model_config.exists():
        logger.error("Config not found: %s", args.model_config)
        sys.exit(1)
    with open(args.model_config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    print(f"Loading models from {args.checkpoint_dir} ...")
    models = load_models(str(args.checkpoint_dir), model_cfg, device=args.device)

    planner_cfg = PlannerConfig(
        n_candidates=args.n_candidates,
        max_steps=args.max_steps,
        history_len=model_cfg.get("action_proposal", {}).get("history_len", 1),
        device=args.device,
        deterministic=True,
        seed=42,
    )
    planner = DMPSPPlanner(
        action_proposal=models["action_proposal"],
        world_model=models["world_model"],
        value_fn=models["value_fn"],
        encoder=models["encoder"],
        cfg=planner_cfg,
    )

    # Evaluate
    results: list[RouteMetrics] = []
    for name, smiles in molecules.items():
        print(f"  Evaluating {name} ({smiles[:40]}...)" if len(smiles) > 40 else f"  Evaluating {name} ({smiles})")
        metrics = evaluate_molecule(name, smiles, planner, objective_weights, args.max_steps)
        results.append(metrics)

    # Print table
    print_summary_table(results)

    # Write JSON
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        out = [
            {
                "name": r.name,
                "smiles": r.smiles,
                "n_steps": r.n_steps,
                "terminated_at_target": r.terminated_at_target,
                "planning_time_s": r.planning_time_s,
                "objective_scores": r.objective_scores,
                "condition_validity": r.condition_validity,
                "reaction_class_ids": r.reaction_class_ids,
                "reaction_class_names": r.reaction_class_names,
                "action_diversity": r.action_diversity,
                "warnings": r.warnings,
            }
            for r in results
        ]
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nResults written to: {args.out_json}")


if __name__ == "__main__":
    main()
