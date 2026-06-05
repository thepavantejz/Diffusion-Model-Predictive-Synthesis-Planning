"""Tests for dmpsp.state — dataclass construction and invariants."""

from __future__ import annotations

import pytest

from dmpsp.state import (
    DMPSPTrajectory,
    ReactionRecord,
    SynthesisAction,
    SynthesisRoute,
    SynthesisState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_action(**kwargs) -> SynthesisAction:
    defaults = dict(
        reaction_class_id=0,
        temperature_norm=0.0,
        pressure_norm=0.0,
        time_norm=0.0,
        solvent_id=0,
        catalyst_id=0,
        reagent_ratio=0.0,
    )
    defaults.update(kwargs)
    return SynthesisAction(**defaults)


def make_state(**kwargs) -> SynthesisState:
    defaults = dict(
        target_smiles="CC(=O)Oc1ccccc1C(=O)O",
        current_smiles="CC(=O)O",
        inventory=[],
        reaction_history=[],
        temperature=298.15,
        pressure=1.0,
        scale=1.0,
        cost_accumulated=0.0,
        step_number=0,
        yield_so_far=1.0,
        purity_so_far=1.0,
    )
    defaults.update(kwargs)
    return SynthesisState(**defaults)


# ---------------------------------------------------------------------------
# SynthesisAction
# ---------------------------------------------------------------------------

def test_synthesis_action_construction():
    action = make_action(reaction_class_id=5, temperature_norm=0.5)
    assert action.reaction_class_id == 5
    assert action.temperature_norm == pytest.approx(0.5)


def test_synthesis_action_default_reagents():
    action = make_action()
    assert action.reagent_smiles == []


def test_synthesis_action_with_reagents():
    action = make_action(reagent_smiles=["CCO", "CC(=O)O"])
    assert len(action.reagent_smiles) == 2


# ---------------------------------------------------------------------------
# SynthesisState
# ---------------------------------------------------------------------------

def test_synthesis_state_construction():
    state = make_state()
    assert state.step_number == 0
    assert state.target_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert state.yield_so_far == pytest.approx(1.0)


def test_synthesis_state_with_history():
    action = make_action()
    state = make_state(reaction_history=[action], step_number=1)
    assert len(state.reaction_history) == 1
    assert state.step_number == 1


# ---------------------------------------------------------------------------
# ReactionRecord
# ---------------------------------------------------------------------------

def test_reaction_record_required_fields():
    rec = ReactionRecord(
        reactant_smiles=["CC(=O)O", "Oc1ccccc1C(=O)O"],
        product_smiles="CC(=O)Oc1ccccc1C(=O)O",
        reaction_class_id=0,
        source="test",
    )
    assert rec.product_smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert len(rec.reactant_smiles) == 2


def test_reaction_record_optional_fields_default_none():
    rec = ReactionRecord(
        reactant_smiles=["CC(=O)O"],
        product_smiles="CCOC(=O)C",
        reaction_class_id=1,
        source="uspto",
    )
    assert rec.yield_percent is None
    assert rec.temperature is None
    assert rec.cost_usd is None


def test_reaction_record_with_full_data():
    rec = ReactionRecord(
        reactant_smiles=["CC(=O)O"],
        product_smiles="CCOC(=O)C",
        reaction_class_id=1,
        source="ord",
        temperature=323.15,
        yield_percent=85.0,
        solvent="CCO",
    )
    assert rec.temperature == pytest.approx(323.15)
    assert rec.yield_percent == pytest.approx(85.0)


# ---------------------------------------------------------------------------
# DMPSPTrajectory
# ---------------------------------------------------------------------------

def test_trajectory_length():
    T = 3
    states = [make_state(step_number=i) for i in range(T + 1)]
    actions = [make_action() for _ in range(T)]
    rewards = [[0.5] * 10 for _ in range(T)]
    traj = DMPSPTrajectory(states, actions, rewards, "CCC", "test")
    assert len(traj) == T


def test_trajectory_state_action_length_invariant():
    """states must have exactly len(actions) + 1 elements."""
    states = [make_state(step_number=i) for i in range(3)]  # 3 states
    actions = [make_action() for _ in range(3)]              # 3 actions — invalid (need 4 states)
    rewards = [[0.5] * 10 for _ in range(3)]
    with pytest.raises(ValueError, match="len\\(states\\) must equal len\\(actions\\) \\+ 1"):
        DMPSPTrajectory(states, actions, rewards, "CCC", "test")


def test_trajectory_rewards_length_invariant():
    T = 3
    states = [make_state() for _ in range(T + 1)]
    actions = [make_action() for _ in range(T)]
    rewards = [[0.5] * 10 for _ in range(T - 1)]  # wrong length
    with pytest.raises(ValueError, match="len\\(rewards_per_objective\\) must equal"):
        DMPSPTrajectory(states, actions, rewards, "CCC", "test")


# ---------------------------------------------------------------------------
# SynthesisRoute
# ---------------------------------------------------------------------------

def test_synthesis_route_construction():
    state = make_state()
    action = make_action()
    route = SynthesisRoute(
        steps=[(state, action)],
        terminal_smiles="CC(=O)O",
        objective_scores={"yield": 0.9, "cost": 0.8},
        total_yield_fraction=0.9,
        total_cost_usd=10.0,
        n_steps=1,
        planning_time_seconds=1.5,
    )
    assert route.n_steps == 1
    assert route.objective_scores["yield"] == pytest.approx(0.9)


def test_synthesis_route_n_steps_invariant():
    state = make_state()
    action = make_action()
    with pytest.raises(ValueError, match="n_steps"):
        SynthesisRoute(
            steps=[(state, action)],
            terminal_smiles="CC(=O)O",
            objective_scores={},
            total_yield_fraction=0.9,
            total_cost_usd=10.0,
            n_steps=5,           # wrong — len(steps) == 1
            planning_time_seconds=1.5,
        )
