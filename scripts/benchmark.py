"""Benchmark DMPSP against baselines on the USPTO-50K test set.

Compares:
  A) DMPSP-Beam  (beam search, deterministic DDIM, default)
  B) DMPSP-MCTS  (MCTS planner, deterministic DDIM)
  C) Random      (random action sequences, value-function scored)

Outputs a CSV table for paper results (Table 1 / Table 2).

Usage:
    python scripts/benchmark.py \
        --checkpoint_dir checkpoints/ \
        --data_dir data/processed/ \
        --device cuda \
        [--n_molecules 100] \
        [--max_steps 5] \
        [--out_csv results/benchmark.csv] \
        [--out_json results/benchmark.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpsp.planner import DMPSPPlanner, PlannerConfig, load_models
from dmpsp.utils import setup_logging, validate_smiles
from dmpsp.value_fn import OBJECTIVE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark result for one method."""
    method: str
    n_evaluated: int
    avg_yield: float
    avg_purity: float
    avg_cost: float
    avg_safety: float
    avg_manufacturability: float
    avg_green_chem: float
    avg_fto_risk: float
    avg_novelty: float
    avg_robustness: float
    avg_supply_avail: float
    avg_steps: float
    avg_planning_time_s: float
    weighted_score: float
    per_molecule: list[dict] = field(default_factory=list)


def _avg_scores(per_molecule: list[dict], key: str) -> float:
    vals = [m[key] for m in per_molecule if key in m]
    return round(sum(vals) / max(len(vals), 1), 4)


def _load_test_smiles(data_dir: Path, n: int) -> list[str]:
    """Load target SMILES from test trajectories."""
    import pickle

    test_path = data_dir / "trajectories_test.pkl"
    if not test_path.exists():
        # Fall back to val set
        test_path = data_dir / "trajectories_val.pkl"
    if not test_path.exists():
        raise FileNotFoundError(f"No test trajectories found in {data_dir}")

    with open(test_path, "rb") as f:
        trajs = pickle.load(f)

    smiles_list: list[str] = []
    for traj in trajs:
        s = None
        if hasattr(traj, "states") and traj.states:
            s = traj.states[0].target_smiles
        elif isinstance(traj, (list, tuple)) and len(traj) > 0:
            item = traj[0]
            if hasattr(item, "target_smiles"):
                s = item.target_smiles
            elif isinstance(item, dict):
                s = item.get("target_smiles")
        if s and validate_smiles(s):
            smiles_list.append(s)
        if len(smiles_list) >= n:
            break

    return smiles_list


def _run_one(
    smiles: str,
    planner: DMPSPPlanner,
    objective_weights: dict[str, float],
    max_steps: int,
) -> Optional[dict]:
    try:
        t0 = time.perf_counter()
        route = planner.plan(smiles, objective_weights, max_steps=max_steps)
        elapsed = time.perf_counter() - t0
        return {
            "smiles": smiles,
            "n_steps": route.n_steps,
            "planning_time_s": round(elapsed, 3),
            **{k: round(v, 4) for k, v in route.objective_scores.items()},
        }
    except Exception as exc:
        logger.warning("Failed on %s: %s", smiles, exc)
        return None


def run_method(
    method_name: str,
    smiles_list: list[str],
    models: dict,
    model_cfg: dict,
    objective_weights: dict[str, float],
    max_steps: int,
    n_candidates: int,
    device: str,
    planner_type: str = "beam",
    deterministic: bool = True,
    seed: Optional[int] = 42,
    random_baseline: bool = False,
) -> BenchmarkResult:
    """Run one method over all test molecules."""
    cfg = PlannerConfig(
        n_candidates=n_candidates,
        max_steps=max_steps,
        planner_type=planner_type,
        beam_width=model_cfg.get("mpc", {}).get("beam_width", 5),
        history_len=model_cfg.get("action_proposal", {}).get("history_len", 1),
        device=device,
        deterministic=deterministic,
        seed=seed,
    )
    planner = DMPSPPlanner(
        action_proposal=models["action_proposal"],
        world_model=models["world_model"],
        value_fn=models["value_fn"],
        encoder=models["encoder"],
        cfg=cfg,
    )

    per_molecule: list[dict] = []
    for i, smiles in enumerate(smiles_list):
        if (i + 1) % 10 == 0:
            logger.info("[%s] %d/%d", method_name, i + 1, len(smiles_list))
        result = _run_one(smiles, planner, objective_weights, max_steps)
        if result is not None:
            per_molecule.append(result)

    n = len(per_molecule)
    weighted = sum(
        sum(m.get(obj, 0.0) * objective_weights.get(obj, 0.0) for obj in OBJECTIVE_NAMES)
        for m in per_molecule
    ) / max(n, 1)

    return BenchmarkResult(
        method=method_name,
        n_evaluated=n,
        avg_yield=_avg_scores(per_molecule, "yield"),
        avg_purity=_avg_scores(per_molecule, "purity"),
        avg_cost=_avg_scores(per_molecule, "cost"),
        avg_safety=_avg_scores(per_molecule, "safety"),
        avg_manufacturability=_avg_scores(per_molecule, "manufacturability"),
        avg_green_chem=_avg_scores(per_molecule, "green_chem"),
        avg_fto_risk=_avg_scores(per_molecule, "fto_risk"),
        avg_novelty=_avg_scores(per_molecule, "novelty"),
        avg_robustness=_avg_scores(per_molecule, "robustness"),
        avg_supply_avail=_avg_scores(per_molecule, "supply_avail"),
        avg_steps=_avg_scores(per_molecule, "n_steps"),
        avg_planning_time_s=_avg_scores(per_molecule, "planning_time_s"),
        weighted_score=round(weighted, 4),
        per_molecule=per_molecule,
    )


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Print paper-style results table."""
    cols = ["Method", "N", "Yield", "Purity", "Cost", "Safety", "Manuf", "GreenChem",
            "Steps", "Time(s)", "WeightedScore"]
    widths = [18, 5, 6, 6, 6, 6, 6, 9, 6, 8, 13]
    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))

    print("\n" + "=" * len(header))
    print("DMPSP Benchmark Results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        row = [
            r.method, str(r.n_evaluated),
            f"{r.avg_yield:.3f}", f"{r.avg_purity:.3f}", f"{r.avg_cost:.3f}",
            f"{r.avg_safety:.3f}", f"{r.avg_manufacturability:.3f}", f"{r.avg_green_chem:.3f}",
            f"{r.avg_steps:.1f}", f"{r.avg_planning_time_s:.2f}",
            f"{r.weighted_score:.4f}",
        ]
        print("  ".join(f"{v:<{w}}" for v, w in zip(row, widths)))

    print("=" * len(header))
    print()
    print("Objective weights used for WeightedScore:")
    print("  yield=0.30, cost=0.20, safety=0.20, manufacturability=0.15, fto_risk=0.15")
    print()
    print("NOTE: Results with <100K training steps reflect model architecture, not")
    print("chemistry quality. Re-run after full training on Kaggle P100 / vast.ai A100.")


def write_csv(results: list[BenchmarkResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "n_evaluated", "avg_yield", "avg_purity", "avg_cost",
        "avg_safety", "avg_manufacturability", "avg_green_chem", "avg_fto_risk",
        "avg_novelty", "avg_robustness", "avg_supply_avail",
        "avg_steps", "avg_planning_time_s", "weighted_score",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in fieldnames})
    print(f"CSV written: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark DMPSP on USPTO-50K test set."
    )
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path(__file__).parent.parent / "data" / "processed")
    parser.add_argument("--model_config", type=Path, default=Path(__file__).parent.parent / "configs" / "model.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--n_molecules", type=int, default=100,
        help="Number of test molecules to evaluate per method. Default: 100.",
    )
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--n_candidates", type=int, default=64)
    parser.add_argument(
        "--methods", nargs="+",
        default=["dmpsp-beam", "dmpsp-mcts", "random"],
        choices=["dmpsp-beam", "dmpsp-mcts", "random"],
        help="Methods to run. Default: all three.",
    )
    parser.add_argument(
        "--weights_json",
        default='{"yield":0.3,"cost":0.2,"safety":0.2,"manufacturability":0.15,"fto_risk":0.15}',
    )
    parser.add_argument("--out_csv", type=Path, default=Path("results/benchmark.csv"))
    parser.add_argument("--out_json", type=Path, default=Path("results/benchmark.json"))
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        objective_weights: dict[str, float] = json.loads(args.weights_json)
    except json.JSONDecodeError as exc:
        logger.error("Invalid --weights_json: %s", exc)
        sys.exit(1)

    if not args.model_config.exists():
        logger.error("Config not found: %s", args.model_config)
        sys.exit(1)
    with open(args.model_config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Load test molecules
    logger.info("Loading test molecules from %s ...", args.data_dir)
    try:
        smiles_list = _load_test_smiles(args.data_dir, args.n_molecules)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    logger.info("Loaded %d test molecules", len(smiles_list))

    # Load models once (shared across methods)
    logger.info("Loading models ...")
    models = load_models(str(args.checkpoint_dir), model_cfg, device=args.device)

    results: list[BenchmarkResult] = []

    method_configs = {
        "dmpsp-beam": dict(planner_type="beam", deterministic=True, seed=42),
        "dmpsp-mcts": dict(planner_type="mcts", deterministic=True, seed=42),
        "random":     dict(planner_type="beam", deterministic=False, seed=None),
    }

    for method in args.methods:
        cfg = method_configs[method]
        logger.info("Running method: %s", method)
        result = run_method(
            method_name=method,
            smiles_list=smiles_list,
            models=models,
            model_cfg=model_cfg,
            objective_weights=objective_weights,
            max_steps=args.max_steps,
            n_candidates=args.n_candidates,
            device=args.device,
            **cfg,
        )
        results.append(result)
        logger.info(
            "%s: weighted_score=%.4f, avg_yield=%.3f, n=%d",
            method, result.weighted_score, result.avg_yield, result.n_evaluated,
        )

    print_results_table(results)
    write_csv(results, args.out_csv)

    # Write full JSON (includes per-molecule results)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "method": r.method,
                    "n_evaluated": r.n_evaluated,
                    "avg_yield": r.avg_yield,
                    "avg_purity": r.avg_purity,
                    "avg_cost": r.avg_cost,
                    "avg_safety": r.avg_safety,
                    "avg_manufacturability": r.avg_manufacturability,
                    "avg_green_chem": r.avg_green_chem,
                    "avg_fto_risk": r.avg_fto_risk,
                    "avg_novelty": r.avg_novelty,
                    "avg_robustness": r.avg_robustness,
                    "avg_supply_avail": r.avg_supply_avail,
                    "avg_steps": r.avg_steps,
                    "avg_planning_time_s": r.avg_planning_time_s,
                    "weighted_score": r.weighted_score,
                    "per_molecule": r.per_molecule,
                }
                for r in results
            ],
            f, indent=2,
        )
    print(f"JSON written: {args.out_json}")


if __name__ == "__main__":
    main()
