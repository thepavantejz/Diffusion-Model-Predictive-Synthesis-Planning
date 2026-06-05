"""Integration tests for DMPSPPlanner — smoke tests with toy model weights."""

from __future__ import annotations

import pytest
import torch

from dmpsp.action_proposal import ActionProposalDiffusion
from dmpsp.encoder import GINEncoder
from dmpsp.planner import DMPSPPlanner, PlannerConfig
from dmpsp.state import SynthesisRoute
from dmpsp.utils import validate_smiles
from dmpsp.value_fn import ValueFunction
from dmpsp.world_model import ChemistryWorldModel


SMALL_AP = dict(
    action_dim=7, state_dim=32, token_dim=32, n_heads=2, mlp_hidden=64,
    n_layers=2, n_diffusion_steps=4, horizon=3, history_len=1, dropout=0.0,
)
SMALL_VF = dict(
    state_dim=32, action_dim=7, token_dim=32, n_heads=2, mlp_hidden=64,
    n_layers=2, n_objectives=10, dropout=0.0,
)


@pytest.fixture(scope="module")
def planner_no_world_model():
    """Planner with real action_proposal and value_fn but stubbed world_model."""
    encoder = GINEncoder(hidden_dim=32, num_layers=2, dropout=0.0)
    action_proposal = ActionProposalDiffusion(**SMALL_AP)
    value_fn = ValueFunction(**SMALL_VF)

    # Stub world model: returns zeros (no ReactionT5 download needed)
    class _StubWorldModel:
        def rollout(self, state_enc, history_enc, actions):
            B, F, _ = actions.shape
            return torch.zeros(B, F, 32)

        def to(self, device):
            return self

        def eval(self):
            return self

    cfg = PlannerConfig(n_candidates=4, max_steps=3, history_len=1, device="cpu")
    return DMPSPPlanner(
        action_proposal=action_proposal,
        world_model=_StubWorldModel(),
        value_fn=value_fn,
        encoder=encoder,
        cfg=cfg,
    )


def test_planner_returns_synthesis_route(planner_no_world_model):
    route = planner_no_world_model.plan(
        target_smiles="CC(=O)O",
        objective_weights={"yield": 0.5, "cost": 0.3, "safety": 0.2},
        max_steps=2,
    )
    assert isinstance(route, SynthesisRoute)


def test_planner_route_has_correct_step_count(planner_no_world_model):
    route = planner_no_world_model.plan(
        target_smiles="CCO",
        objective_weights={"yield": 1.0},
        max_steps=3,
    )
    assert route.n_steps == len(route.steps)
    assert route.n_steps <= 3


def test_planner_route_objective_scores_have_all_keys(planner_no_world_model):
    from dmpsp.value_fn import OBJECTIVE_NAMES
    route = planner_no_world_model.plan(
        target_smiles="CCO",
        objective_weights={"yield": 1.0},
        max_steps=2,
    )
    for name in OBJECTIVE_NAMES:
        assert name in route.objective_scores


def test_planner_planning_time_is_positive(planner_no_world_model):
    route = planner_no_world_model.plan(
        target_smiles="CC",
        objective_weights={"yield": 1.0},
        max_steps=1,
    )
    assert route.planning_time_seconds > 0


def test_planner_invalid_smiles_raises():
    encoder = GINEncoder(hidden_dim=32, num_layers=2, dropout=0.0)
    action_proposal = ActionProposalDiffusion(**SMALL_AP)
    value_fn = ValueFunction(**SMALL_VF)

    class _Stub:
        def rollout(self, s, h, a):
            return torch.zeros(a.shape[0], a.shape[1], 32)
        def to(self, d): return self
        def eval(self): return self

    cfg = PlannerConfig(n_candidates=4, max_steps=2, device="cpu")
    planner = DMPSPPlanner(action_proposal, _Stub(), value_fn, encoder, cfg)

    with pytest.raises(ValueError):
        planner.plan("NOT_A_VALID_SMILES", {"yield": 1.0})


def test_planner_weights_from_dict_missing_keys(planner_no_world_model):
    """Planner should handle weights with only a subset of objectives."""
    route = planner_no_world_model.plan(
        target_smiles="CCC",
        objective_weights={"yield": 1.0},  # only one objective
        max_steps=2,
    )
    assert isinstance(route, SynthesisRoute)
