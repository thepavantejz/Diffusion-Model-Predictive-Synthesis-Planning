"""Core data structures for DMPSP.

All planning state, actions, trajectories, and routes are defined here.
Pure data containers — no ML logic, no external dependencies beyond stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SynthesisAction:
    """A single synthesis reaction step with conditions."""

    reaction_class_id: int          # index into reaction taxonomy (num_classes from config)
    temperature_norm: float         # normalized to [-1, 1] relative to config min/max
    pressure_norm: float            # normalized to [-1, 1]
    time_norm: float                # normalized to [-1, 1]
    solvent_id: int                 # index into solvent vocabulary
    catalyst_id: int                # index into catalyst vocabulary
    reagent_ratio: float            # normalized stoichiometric ratio [-1, 1]
    reagent_smiles: list[str] = field(default_factory=list)  # for logging/display only


@dataclass
class SynthesisState:
    """Full synthesis context at a given planning step.

    target_smiles is fixed throughout planning — it is the molecule we want to make.
    current_smiles changes at each step as reactions are executed.
    """

    target_smiles: str
    current_smiles: str
    inventory: list[str]                        # canonical SMILES of available reagents
    reaction_history: list[SynthesisAction]     # actions executed before this state
    temperature: float                          # current process temperature (K)
    pressure: float                             # current pressure (atm)
    scale: float                                # batch scale (g)
    cost_accumulated: float                     # total cost so far (USD)
    step_number: int                            # 0-indexed synthesis step
    yield_so_far: float                         # cumulative yield fraction [0, 1]
    purity_so_far: float                        # current product purity [0, 1]


@dataclass
class ReactionRecord:
    """Standardized reaction record from any data source.

    Used as the common intermediate format between raw data loaders
    (CSV, USPTO, ORD) and TrajectoryBuilder.
    """

    reactant_smiles: list[str]          # list of reactant canonical SMILES
    product_smiles: str                 # canonical SMILES of main product
    reaction_class_id: int              # integer class from tokenizer
    source: str                         # "csv" | "uspto" | "ord" | "chembl"
    temperature: Optional[float] = None     # K
    pressure: Optional[float] = None        # atm
    time_hours: Optional[float] = None
    solvent: Optional[str] = None           # SMILES or name
    catalyst: Optional[str] = None
    reagent_ratio: Optional[float] = None
    yield_percent: Optional[float] = None   # [0, 100]
    purity_percent: Optional[float] = None  # [0, 100]
    cost_usd: Optional[float] = None        # per-gram cost of starting materials
    metadata: dict = field(default_factory=dict)


@dataclass
class DMPSPTrajectory:
    """Training trajectory: full sequence of states, actions, and per-objective rewards.

    rewards_per_objective[t] is a list of N_OBJECTIVES floats, each in [0, 1].
    len(states) == len(actions) + 1  (states include the initial state)
    """

    states: list[SynthesisState]
    actions: list[SynthesisAction]
    rewards_per_objective: list[list[float]]    # shape [T, N_OBJECTIVES]
    terminal_smiles: str
    source: str

    def __len__(self) -> int:
        return len(self.actions)

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions) + 1:
            raise ValueError(
                f"len(states) must equal len(actions) + 1. "
                f"Got len(states)={len(self.states)}, len(actions)={len(self.actions)}."
            )
        if len(self.rewards_per_objective) != len(self.actions):
            raise ValueError(
                f"len(rewards_per_objective) must equal len(actions). "
                f"Got {len(self.rewards_per_objective)} vs {len(self.actions)}."
            )


@dataclass
class SynthesisRoute:
    """Output of DMPSPPlanner.plan() — a ranked synthesis route with all objective scores."""

    steps: list[tuple[SynthesisState, SynthesisAction]]
    terminal_smiles: str
    objective_scores: dict[str, float]      # 10 objectives, each in [0, 1]
    total_yield_fraction: float             # product of per-step yields
    total_cost_usd: float
    n_steps: int
    planning_time_seconds: float

    def __post_init__(self) -> None:
        if self.n_steps != len(self.steps):
            raise ValueError(
                f"n_steps={self.n_steps} does not match len(steps)={len(self.steps)}."
            )
