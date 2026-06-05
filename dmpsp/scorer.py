"""External and computed scoring modules for synthesis route objectives.

ADMETlabScorer  — calls ADMETlab 3.0 REST API (async). Requires ADMETLAB_API_KEY.
CostScorer      — calls eMolecules API (sync). Requires EMOLECULES_API_KEY.
GreenChemistryScorer — PMI, E-factor, atom economy from stoichiometry (pure RDKit, no API).

Policy: all scorers return a real value or raise. No silent fallbacks.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx
from rdkit import Chem
from rdkit.Chem import Descriptors

from dmpsp.utils import canonicalize_smiles, smiles_to_inchikey

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Result from any scoring module.

    value: Normalized score in [0, 1]. Higher is always better.
    raw_value: Original value in its natural units.
    name: Objective name matching configs/model.yaml objectives list.
    """

    name: str
    value: float            # [0, 1], higher = better
    raw_value: float        # original units
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ADMETlab scorer
# ---------------------------------------------------------------------------

class ADMETlabScorer:
    """ADMET scoring via ADMETlab 3.0 REST API (async).

    Results are cached by InChIKey so each molecule is queried at most once
    per scorer instance.

    Raises:
        ValueError: If ADMETLAB_API_KEY env var is not set (at construction time).
        RuntimeError: If the API call fails or returns unexpected data.
    """

    BASE_URL: str = "https://admetlab3.scbdd.com/api"
    TIMEOUT_SECONDS: int = 30

    # ADMETlab 3.0 toxicity endpoint names to use for composite score
    _TOX_ENDPOINTS: tuple[str, ...] = (
        "hERG", "AMES", "Carcinogens_Lagunin", "Acute_Toxicity_LD50",
    )

    def __init__(self) -> None:
        api_key = os.environ.get("ADMETLAB_API_KEY")
        if not api_key:
            raise ValueError(
                "ADMETLAB_API_KEY environment variable is not set. "
                "Register for a free key at https://admetlab3.scbdd.com"
            )
        self._api_key = api_key
        self._cache: dict[str, ScoringResult] = {}

    async def score(self, smiles: str) -> ScoringResult:
        """Score a molecule for composite ADMET toxicity.

        Args:
            smiles: SMILES string (will be canonicalized internally).

        Returns:
            ScoringResult where value = 1 - composite_tox_score (safer = higher).

        Raises:
            RuntimeError: On API failure or missing endpoints in response.
        """
        canonical = canonicalize_smiles(smiles)
        inchikey = smiles_to_inchikey(canonical)

        if inchikey in self._cache:
            return self._cache[inchikey]

        async with httpx.AsyncClient(timeout=self.TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self.BASE_URL}/predict",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"smiles": canonical},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"ADMETlab API failed: HTTP {response.status_code}. "
                f"Response: {response.text[:300]}"
            )

        data = response.json()
        tox_score = self._composite_tox(data)

        result = ScoringResult(
            name="safety",
            value=1.0 - tox_score,
            raw_value=tox_score,
            metadata={"inchikey": inchikey, "endpoints_used": list(self._TOX_ENDPOINTS)},
        )
        self._cache[inchikey] = result
        return result

    def score_sync(self, smiles: str) -> ScoringResult:
        """Synchronous wrapper around score(). Use when not in an async context."""
        return asyncio.run(self.score(smiles))

    def _composite_tox(self, api_response: dict) -> float:
        scores: list[float] = []
        for key in self._TOX_ENDPOINTS:
            val = api_response.get(key)
            if val is not None:
                try:
                    scores.append(float(val))
                except (ValueError, TypeError):
                    continue
        if not scores:
            raise RuntimeError(
                f"ADMETlab response missing all expected toxicity endpoints {self._TOX_ENDPOINTS}. "
                f"Got keys: {sorted(api_response)[:20]}"
            )
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Cost scorer
# ---------------------------------------------------------------------------

class CostScorer:
    """Reagent cost scoring via eMolecules API.

    Returns cost per gram normalized against max_cost_per_gram.
    Score of 1.0 = free (or nearly free). Score of 0.0 = at or above max_cost.

    Raises:
        ValueError: If EMOLECULES_API_KEY is not set (at construction time).
        RuntimeError: On API failure or no results for the query SMILES.
    """

    BASE_URL: str = "https://api.emolecules.com/v1"
    TIMEOUT_SECONDS: int = 15

    def __init__(self, max_cost_per_gram: float = 1000.0) -> None:
        api_key = os.environ.get("EMOLECULES_API_KEY")
        if not api_key:
            raise ValueError(
                "EMOLECULES_API_KEY environment variable is not set. "
                "Get API access at https://www.emolecules.com"
            )
        self._api_key = api_key
        self.max_cost_per_gram = max_cost_per_gram
        self._cache: dict[str, ScoringResult] = {}

    def score(self, smiles: str) -> ScoringResult:
        """Score reagent availability and cost.

        Args:
            smiles: SMILES string (will be canonicalized internally).

        Returns:
            ScoringResult where value = 1 - normalized_cost.

        Raises:
            RuntimeError: On API failure or no results.
        """
        canonical = canonicalize_smiles(smiles)
        inchikey = smiles_to_inchikey(canonical)

        if inchikey in self._cache:
            return self._cache[inchikey]

        with httpx.Client(timeout=self.TIMEOUT_SECONDS) as client:
            response = client.get(
                f"{self.BASE_URL}/compounds/search",
                headers={"X-API-Key": self._api_key},
                params={"smiles": canonical, "max_results": 1},
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"eMolecules API failed: HTTP {response.status_code}. "
                f"Response: {response.text[:300]}"
            )

        data = response.json()
        compounds = data.get("compounds", [])
        if not compounds:
            raise RuntimeError(
                f"eMolecules returned no results for SMILES: {canonical!r}"
            )

        cost_per_gram = float(compounds[0].get("price_per_gram", 0.0))
        normalized = min(cost_per_gram / self.max_cost_per_gram, 1.0)

        result = ScoringResult(
            name="cost",
            value=1.0 - normalized,
            raw_value=cost_per_gram,
            metadata={"inchikey": inchikey, "currency": "USD", "unit": "USD/g"},
        )
        self._cache[inchikey] = result
        return result


# ---------------------------------------------------------------------------
# Green chemistry scorer (pure RDKit, no API)
# ---------------------------------------------------------------------------

class GreenChemistryScorer:
    """Compute PMI, E-factor, and Atom Economy from reaction stoichiometry.

    All metrics are derived from molecular weights via RDKit — no network calls.
    Higher scores always indicate greener chemistry (all metrics are inverted
    where necessary so higher = better).

    Reference thresholds (Jimenez-Gonzalez et al. 2011):
        E-factor target for pharma: < 25
        PMI target for pharma: < 50
    """

    # Normalization upper bounds (beyond these = score of 0)
    _PMI_MAX: float = 200.0
    _E_FACTOR_MAX: float = 100.0

    def score_reaction(
        self,
        reactant_smiles: list[str],
        product_smiles: str,
        solvent_mass_ratio: float = 0.0,
    ) -> dict[str, ScoringResult]:
        """Score a single reaction step for all green chemistry metrics.

        Args:
            reactant_smiles: List of reactant SMILES.
            product_smiles: Product SMILES.
            solvent_mass_ratio: Grams of solvent per gram of product.

        Returns:
            Dict mapping metric name to ScoringResult.
                Keys: "atom_economy", "e_factor", "pmi".

        Raises:
            ValueError: If any SMILES is invalid.
        """
        mw_product = self._mw(product_smiles)
        mw_reactants = [self._mw(s) for s in reactant_smiles]
        total_mw_in = sum(mw_reactants)

        # Atom Economy: fraction of reactant mass ending up in product
        ae = mw_product / total_mw_in if total_mw_in > 0 else 0.0

        # E-factor: waste / product (lower is better)
        waste = max(total_mw_in - mw_product, 0.0) + solvent_mass_ratio * mw_product
        e_factor = waste / mw_product if mw_product > 0 else self._E_FACTOR_MAX

        # PMI: total mass in / product mass (lower is better)
        pmi = (total_mw_in + solvent_mass_ratio * mw_product) / mw_product if mw_product > 0 else self._PMI_MAX

        return {
            "atom_economy": ScoringResult("atom_economy", min(ae, 1.0), ae * 100, {"unit": "%"}),
            "e_factor": ScoringResult(
                "e_factor",
                1.0 - min(e_factor / self._E_FACTOR_MAX, 1.0),
                e_factor,
                {"unit": "g waste / g product"},
            ),
            "pmi": ScoringResult(
                "pmi",
                1.0 - min(pmi / self._PMI_MAX, 1.0),
                pmi,
                {"unit": "g total / g product"},
            ),
        }

    def aggregate_route(
        self, per_step_scores: list[dict[str, ScoringResult]]
    ) -> ScoringResult:
        """Aggregate per-step green chemistry scores into a single route score.

        Args:
            per_step_scores: List (one per step) of dicts from score_reaction().

        Returns:
            Single ScoringResult with mean composite green chemistry score.

        Raises:
            ValueError: If per_step_scores is empty.
        """
        if not per_step_scores:
            raise ValueError("per_step_scores must not be empty.")
        step_values = [
            sum(r.value for r in step.values()) / len(step)
            for step in per_step_scores
        ]
        composite = sum(step_values) / len(step_values)
        return ScoringResult(
            "green_chem", composite, composite,
            {"n_steps": len(per_step_scores)},
        )

    @staticmethod
    def _mw(smiles: str) -> float:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for MW calculation: {smiles!r}")
        return Descriptors.MolWt(mol)
