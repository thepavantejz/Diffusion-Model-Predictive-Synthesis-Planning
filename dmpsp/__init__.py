"""DMPSP: Diffusion Model Predictive Synthesis Planning.

A controllable synthesis planning world model for pharmaceutical route discovery.
Based on D-MPC (arXiv:2410.05364, TMLR 2025).

Quick start::

    from dmpsp import DMPSPPlanner, load_models

    models = load_models(checkpoint_dir="checkpoints/")
    planner = DMPSPPlanner(**models)
    route = planner.plan(
        target_smiles="CC(=O)Oc1ccccc1C(=O)O",
        objective_weights={"yield": 0.4, "cost": 0.3, "safety": 0.3},
    )
    print(route)
"""

from dmpsp.state import (
    DMPSPTrajectory,
    ReactionRecord,
    SynthesisAction,
    SynthesisRoute,
    SynthesisState,
)
from dmpsp.planner import DMPSPPlanner, PlannerConfig, load_models

__version__ = "0.1.0"

__all__ = [
    "SynthesisState",
    "SynthesisAction",
    "SynthesisRoute",
    "DMPSPTrajectory",
    "ReactionRecord",
    "DMPSPPlanner",
    "PlannerConfig",
    "load_models",
]
