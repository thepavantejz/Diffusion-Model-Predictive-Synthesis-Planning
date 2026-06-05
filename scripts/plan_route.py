"""Inference CLI: find synthesis routes for a target molecule.

Usage:
    python scripts/plan_route.py \
        --smiles "CC(=O)Oc1ccccc1C(=O)O" \
        --weights_json '{"yield":0.3,"cost":0.2,"safety":0.2,"manufacturability":0.15,"fto_risk":0.15}' \
        --model_config configs/model.yaml \
        --checkpoint_dir checkpoints/ \
        --device cpu \
        --max_steps 5

Output: JSON to stdout with SynthesisRoute details.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from dmpsp.encoder import build_encoder
from dmpsp.planner import DMPSPPlanner, PlannerConfig, load_models
from dmpsp.utils import setup_logging, validate_smiles

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find synthesis routes for a target molecule using DMPSP."
    )
    parser.add_argument(
        "--smiles", required=True,
        help="Target molecule SMILES string.",
    )
    parser.add_argument(
        "--weights_json",
        default='{"yield":0.3,"cost":0.2,"safety":0.2,"manufacturability":0.15,"fto_risk":0.15}',
        help="JSON string mapping objective name to weight. Default: balanced weights.",
    )
    parser.add_argument(
        "--model_config", type=Path, default=Path("configs/model.yaml"),
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=Path, required=True,
        help="Directory containing model checkpoints.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device to run inference on (cuda, cpu). Default: cpu.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=10,
        help="Maximum number of synthesis steps. Default: 10.",
    )
    parser.add_argument(
        "--n_candidates", type=int, default=64,
        help="Number of candidate action sequences to sample per step. Default: 64.",
    )
    parser.add_argument(
        "--log_level", default="WARNING",
        help="Logging level. Default: WARNING (quiet output).",
    )
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Validate inputs
    if not validate_smiles(args.smiles):
        logger.error("Invalid SMILES: %r", args.smiles)
        sys.exit(1)

    try:
        objective_weights: dict[str, float] = json.loads(args.weights_json)
    except json.JSONDecodeError as exc:
        logger.error("Invalid --weights_json: %s", exc)
        sys.exit(1)

    if not args.model_config.exists():
        logger.error("Model config not found: %s", args.model_config)
        sys.exit(1)

    with open(args.model_config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Load models
    models = load_models(str(args.checkpoint_dir), model_cfg, device=args.device)

    planner_cfg = PlannerConfig(
        n_candidates=args.n_candidates,
        max_steps=args.max_steps,
        history_len=model_cfg.get("action_proposal", {}).get("history_len", 1),
        device=args.device,
    )
    planner = DMPSPPlanner(
        action_proposal=models["action_proposal"],
        world_model=models["world_model"],
        value_fn=models["value_fn"],
        encoder=models["encoder"],
        cfg=planner_cfg,
    )

    # Run planning
    logger.info("Planning synthesis route for: %s", args.smiles)
    route = planner.plan(args.smiles, objective_weights, max_steps=args.max_steps)

    # Output as JSON
    output = {
        "target_smiles": args.smiles,
        "terminal_smiles": route.terminal_smiles,
        "n_steps": route.n_steps,
        "total_yield_fraction": round(route.total_yield_fraction, 4),
        "total_cost_usd": round(route.total_cost_usd, 2),
        "planning_time_seconds": round(route.planning_time_seconds, 2),
        "objective_scores": {k: round(v, 4) for k, v in route.objective_scores.items()},
        "steps": [
            {
                "step": i + 1,
                "current_smiles": state.current_smiles,
                "reaction_class_id": action.reaction_class_id,
                "temperature_norm": round(action.temperature_norm, 3),
                "pressure_norm": round(action.pressure_norm, 3),
                "time_norm": round(action.time_norm, 3),
                "solvent_id": action.solvent_id,
                "catalyst_id": action.catalyst_id,
                "reagent_ratio": round(action.reagent_ratio, 3),
            }
            for i, (state, action) in enumerate(route.steps)
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
