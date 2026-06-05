"""Tests for dmpsp.scorer — green chemistry is always available; API tests skip if no key."""

from __future__ import annotations

import os

import pytest

from dmpsp.scorer import GreenChemistryScorer, ScoringResult


# ---------------------------------------------------------------------------
# GreenChemistryScorer (pure RDKit, always available)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def scorer() -> GreenChemistryScorer:
    return GreenChemistryScorer()


def test_green_chemistry_score_reaction_returns_three_metrics(scorer):
    result = scorer.score_reaction(
        reactant_smiles=["CC(=O)O", "Oc1ccccc1C(=O)O"],
        product_smiles="CC(=O)Oc1ccccc1C(=O)O",
    )
    assert set(result.keys()) == {"atom_economy", "e_factor", "pmi"}


def test_green_chemistry_all_values_in_zero_one(scorer):
    result = scorer.score_reaction(
        reactant_smiles=["CC(=O)O"],
        product_smiles="CCO",
    )
    for metric, r in result.items():
        assert isinstance(r, ScoringResult), f"{metric} should be ScoringResult"
        assert 0.0 <= r.value <= 1.0, f"{metric} value {r.value} not in [0, 1]"


def test_atom_economy_perfect_reaction(scorer):
    """When product MW equals reactant MW (impossible in practice, but tests normalization)."""
    result = scorer.score_reaction(
        reactant_smiles=["CCO"],
        product_smiles="CCO",
        solvent_mass_ratio=0.0,
    )
    assert result["atom_economy"].value == pytest.approx(1.0, abs=1e-3)


def test_green_chemistry_invalid_smiles_raises(scorer):
    with pytest.raises(ValueError, match="Invalid SMILES"):
        scorer.score_reaction(["NOT_SMILES"], "CCO")


def test_green_chemistry_aggregate_route(scorer):
    step_results = [
        scorer.score_reaction(["CC(=O)O"], "CCO"),
        scorer.score_reaction(["CCO", "O"], "CC(O)O"),
    ]
    agg = scorer.aggregate_route(step_results)
    assert isinstance(agg, ScoringResult)
    assert agg.name == "green_chem"
    assert 0.0 <= agg.value <= 1.0


def test_aggregate_empty_raises(scorer):
    with pytest.raises(ValueError, match="must not be empty"):
        scorer.aggregate_route([])


def test_scoring_result_fields():
    result = ScoringResult(name="test", value=0.75, raw_value=42.0, metadata={"unit": "g"})
    assert result.name == "test"
    assert result.value == pytest.approx(0.75)
    assert result.raw_value == pytest.approx(42.0)
    assert result.metadata["unit"] == "g"


# ---------------------------------------------------------------------------
# ADMETlabScorer — skipped if API key not set
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("ADMETLAB_API_KEY"),
    reason="ADMETLAB_API_KEY not set — skipping live API test",
)
def test_admetlab_scorer_returns_valid_result():
    from dmpsp.scorer import ADMETlabScorer
    scorer_api = ADMETlabScorer()
    result = scorer_api.score_sync("CC(=O)O")  # acetic acid
    assert isinstance(result, ScoringResult)
    assert 0.0 <= result.value <= 1.0
    assert result.name == "safety"


def test_admetlab_scorer_raises_without_key(monkeypatch):
    """Without API key, construction must raise ValueError."""
    monkeypatch.delenv("ADMETLAB_API_KEY", raising=False)
    from dmpsp.scorer import ADMETlabScorer
    with pytest.raises(ValueError, match="ADMETLAB_API_KEY"):
        ADMETlabScorer()


def test_cost_scorer_raises_without_key(monkeypatch):
    """Without API key, construction must raise ValueError."""
    monkeypatch.delenv("EMOLECULES_API_KEY", raising=False)
    from dmpsp.scorer import CostScorer
    with pytest.raises(ValueError, match="EMOLECULES_API_KEY"):
        CostScorer()
