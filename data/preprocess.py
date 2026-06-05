"""Preprocessing pipeline: ReactionRecord → DMPSPTrajectory.

build_trajectories() is the main entry point. It converts any iterable of
ReactionRecord objects (from any loader) into DMPSPTrajectory training data.

Single-step reactions become 1-action trajectories.
Multi-step routes (when available) become longer trajectories.
"""

from __future__ import annotations

import logging
import pickle
import random
from pathlib import Path
from typing import Iterable, Optional

from dmpsp.state import (
    DMPSPTrajectory,
    ReactionRecord,
    SynthesisAction,
    SynthesisState,
)

logger = logging.getLogger(__name__)

# Default normalization ranges (overridden by data config)
_DEFAULT_TEMP_RANGE = (200.0, 450.0)    # K
_DEFAULT_PRESSURE_RANGE = (0.1, 20.0)  # atm
_DEFAULT_TIME_RANGE = (0.1, 72.0)      # hours
_DEFAULT_RATIO_RANGE = (0.5, 5.0)      # stoichiometric ratio


def build_trajectories(
    records: Iterable[ReactionRecord],
    data_cfg: dict,
    objective_weights: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> list[DMPSPTrajectory]:
    """Convert an iterable of ReactionRecord into a list of DMPSPTrajectory.

    Each ReactionRecord becomes a single-step trajectory (T=1). Multi-step
    trajectories can be constructed by chaining records when the product of
    one record matches the reactant of the next (handled by group_into_routes()).

    Reward computation uses objective_weights to scale per-objective scores.
    If objective_weights is None, all objectives receive equal weight.

    Args:
        records: Iterable of ReactionRecord from any loader.
        data_cfg: Data config dict (from configs/data.yaml).
        objective_weights: Dict mapping objective name to weight. All missing
                           objectives default to 0.0.
        seed: Random seed (used for any stochastic preprocessing).

    Returns:
        List of DMPSPTrajectory ready for SynthesisDataset.
    """
    random.seed(seed)
    n_objectives = 10   # fixed: must match model config

    if objective_weights is None:
        objective_weights = {obj: 1.0 / n_objectives for obj in _OBJECTIVE_NAMES}

    norm = _Normalizer(data_cfg)
    trajectories: list[DMPSPTrajectory] = []

    for record in records:
        traj = _record_to_trajectory(record, norm, n_objectives)
        if traj is not None:
            trajectories.append(traj)

    logger.info("Built %d trajectories from records.", len(trajectories))
    return trajectories


def split_trajectories(
    trajectories: list[DMPSPTrajectory],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int = 42,
) -> tuple[list[DMPSPTrajectory], list[DMPSPTrajectory], list[DMPSPTrajectory]]:
    """Split trajectories into train/val/test sets.

    Args:
        trajectories: Full list of trajectories.
        train_frac: Fraction for training (e.g. 0.90).
        val_frac: Fraction for validation.
        test_frac: Fraction for test. Must sum to 1.0 with train_frac + val_frac.
        seed: Shuffle seed.

    Returns:
        Tuple of (train, val, test) trajectory lists.

    Raises:
        ValueError: If fractions do not sum to 1.0.
    """
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(
            f"train_frac + val_frac + test_frac must sum to 1.0, got {total_frac:.4f}."
        )

    shuffled = trajectories.copy()
    random.seed(seed)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    logger.info(
        "Split: %d train / %d val / %d test (total %d)",
        len(train), len(val), len(test), n,
    )
    return train, val, test


def save_trajectories(trajectories: list[DMPSPTrajectory], path: Path) -> None:
    """Serialize trajectories to a pickle file.

    Args:
        trajectories: List of DMPSPTrajectory.
        path: Output .pkl file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(trajectories, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %d trajectories to %s", len(trajectories), path)


def load_trajectories(path: Path) -> list[DMPSPTrajectory]:
    """Load trajectories from a pickle file.

    Args:
        path: Path to .pkl file written by save_trajectories().

    Returns:
        List of DMPSPTrajectory.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {path}")
    with open(path, "rb") as f:
        trajectories = pickle.load(f)
    logger.info("Loaded %d trajectories from %s", len(trajectories), path)
    return trajectories


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OBJECTIVE_NAMES: list[str] = [
    "yield", "purity", "cost", "novelty", "fto_risk",
    "green_chem", "manufacturability", "safety", "robustness", "supply_avail",
]


def _record_to_trajectory(
    record: ReactionRecord,
    norm: "_Normalizer",
    n_objectives: int,
) -> Optional[DMPSPTrajectory]:
    """Convert a single ReactionRecord to a 1-step DMPSPTrajectory."""
    action = norm.record_to_action(record)
    if action is None:
        return None

    initial_state = SynthesisState(
        target_smiles=record.product_smiles,
        current_smiles=record.reactant_smiles[0] if record.reactant_smiles else record.product_smiles,
        inventory=record.reactant_smiles[1:] if len(record.reactant_smiles) > 1 else [],
        reaction_history=[],
        temperature=record.temperature or 298.15,
        pressure=record.pressure or 1.0,
        scale=1.0,
        cost_accumulated=record.cost_usd or 0.0,
        step_number=0,
        yield_so_far=1.0,
        purity_so_far=1.0,
    )

    final_state = SynthesisState(
        target_smiles=record.product_smiles,
        current_smiles=record.product_smiles,
        inventory=[],
        reaction_history=[action],
        temperature=record.temperature or 298.15,
        pressure=record.pressure or 1.0,
        scale=1.0,
        cost_accumulated=record.cost_usd or 0.0,
        step_number=1,
        yield_so_far=(record.yield_percent or 100.0) / 100.0,
        purity_so_far=(record.purity_percent or 100.0) / 100.0,
    )

    rewards = [_compute_rewards(record, n_objectives)]

    return DMPSPTrajectory(
        states=[initial_state, final_state],
        actions=[action],
        rewards_per_objective=rewards,
        terminal_smiles=record.product_smiles,
        source=record.source,
    )


def _compute_rewards(record: ReactionRecord, n_objectives: int) -> list[float]:
    """Compute per-objective reward vector for a ReactionRecord.

    Objectives without data get a neutral reward of 0.5.
    """
    rewards = [0.5] * n_objectives  # neutral default

    # yield (index 0)
    if record.yield_percent is not None:
        rewards[0] = min(record.yield_percent / 100.0, 1.0)

    # purity (index 1)
    if record.purity_percent is not None:
        rewards[1] = min(record.purity_percent / 100.0, 1.0)

    # cost (index 2): lower cost = higher reward; normalize to [0, 1]
    if record.cost_usd is not None:
        rewards[2] = max(0.0, 1.0 - record.cost_usd / 1000.0)

    # objectives 3-9 default to 0.5 (unknown) — updated by scorer modules

    return rewards


class _Normalizer:
    """Normalize ReactionRecord continuous fields to [-1, 1] action space."""

    def __init__(self, cfg: dict) -> None:
        self.temp_min = cfg.get("temp_min_k", _DEFAULT_TEMP_RANGE[0])
        self.temp_max = cfg.get("temp_max_k", _DEFAULT_TEMP_RANGE[1])
        self.pressure_min = cfg.get("pressure_min_atm", _DEFAULT_PRESSURE_RANGE[0])
        self.pressure_max = cfg.get("pressure_max_atm", _DEFAULT_PRESSURE_RANGE[1])
        self.time_min = cfg.get("time_min_h", _DEFAULT_TIME_RANGE[0])
        self.time_max = cfg.get("time_max_h", _DEFAULT_TIME_RANGE[1])
        self.ratio_min = cfg.get("reagent_ratio_min", _DEFAULT_RATIO_RANGE[0])
        self.ratio_max = cfg.get("reagent_ratio_max", _DEFAULT_RATIO_RANGE[1])

    def _norm(self, value: float, lo: float, hi: float) -> float:
        """Map value from [lo, hi] to [-1, 1], clipped."""
        if hi <= lo:
            return 0.0
        return max(-1.0, min(1.0, 2.0 * (value - lo) / (hi - lo) - 1.0))

    def record_to_action(self, record: ReactionRecord) -> Optional[SynthesisAction]:
        """Convert a ReactionRecord into a SynthesisAction. Returns None if invalid."""
        return SynthesisAction(
            reaction_class_id=record.reaction_class_id,
            temperature_norm=self._norm(
                record.temperature or (self.temp_min + self.temp_max) / 2,
                self.temp_min, self.temp_max,
            ),
            pressure_norm=self._norm(
                record.pressure or (self.pressure_min + self.pressure_max) / 2,
                self.pressure_min, self.pressure_max,
            ),
            time_norm=self._norm(
                record.time_hours or (self.time_min + self.time_max) / 2,
                self.time_min, self.time_max,
            ),
            solvent_id=0,       # placeholder until vocabulary is built during preprocess
            catalyst_id=0,
            reagent_ratio=self._norm(
                record.reagent_ratio or (self.ratio_min + self.ratio_max) / 2,
                self.ratio_min, self.ratio_max,
            ),
            reagent_smiles=record.reactant_smiles,
        )
