"""DMPSPPlanner: MPC loop combining all three trained components.

Implements Algorithm 2 from D-MPC paper, adapted for synthesis planning.

At each step:
  1. Sample N candidate action sequences from ActionProposalDiffusion
  2. Predict N outcome trajectories via ChemistryWorldModel
  3. Score all N trajectories via ValueFunction with runtime weights
  4. Execute the first action of the best candidate
  5. Observe real next state, update history, replan

Runtime objective weights can be changed between calls to plan() without
any model retraining — this is the core D-MPC property.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from dmpsp.action_proposal import ActionProposalDiffusion
from dmpsp.encoder import MolecularEncoder
from dmpsp.state import SynthesisAction, SynthesisRoute, SynthesisState
from dmpsp.utils import canonicalize_smiles, validate_smiles
from dmpsp.value_fn import ValueFunction, OBJECTIVE_NAMES
from dmpsp.world_model import ChemistryWorldModel

logger = logging.getLogger(__name__)


@dataclass
class PlannerConfig:
    """Configuration for DMPSPPlanner."""

    n_candidates: int = 64          # N — action sequences to sample
    max_steps: int = 10             # maximum synthesis steps
    planner_type: str = "beam"      # "beam" | "mcts"
    beam_width: int = 5             # beam search width
    history_len: int = 1            # H — history context length
    device: str = "cpu"


class DMPSPPlanner:
    """MPC planner combining ActionProposalDiffusion, ChemistryWorldModel, ValueFunction.

    Args:
        action_proposal: Trained ActionProposalDiffusion model.
        world_model: Trained ChemistryWorldModel.
        value_fn: Trained ValueFunction.
        encoder: Molecular encoder for SMILES → tensor conversion.
        cfg: PlannerConfig.
    """

    def __init__(
        self,
        action_proposal: ActionProposalDiffusion,
        world_model: ChemistryWorldModel,
        value_fn: ValueFunction,
        encoder: MolecularEncoder,
        cfg: PlannerConfig,
    ) -> None:
        self.action_proposal = action_proposal
        self.world_model = world_model
        self.value_fn = value_fn
        self.encoder = encoder
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Move models to device
        self.action_proposal.to(self.device)
        self.world_model.to(self.device)
        self.value_fn.to(self.device)
        self.encoder.to(self.device)

        # Set eval mode
        for m in [self.action_proposal, self.world_model, self.value_fn, self.encoder]:
            m.eval()

    @torch.no_grad()
    def plan(
        self,
        target_smiles: str,
        objective_weights: dict[str, float],
        starting_smiles: Optional[str] = None,
        inventory: Optional[list[str]] = None,
        max_steps: Optional[int] = None,
    ) -> SynthesisRoute:
        """Find an optimal synthesis route for target_smiles.

        Args:
            target_smiles: SMILES string of the target molecule to synthesize.
            objective_weights: Runtime objective weights. Keys must be from OBJECTIVE_NAMES.
                               Need not sum to 1. Missing objectives are weighted 0.
            starting_smiles: Optional starting material SMILES. If None, planning starts
                             from the target itself (retrosynthesis mode).
            inventory: List of available reagent SMILES.
            max_steps: Override cfg.max_steps for this call.

        Returns:
            SynthesisRoute with steps, scores, and metadata.

        Raises:
            ValueError: If target_smiles is invalid.
        """
        target_smiles = canonicalize_smiles(target_smiles)
        current_smiles = starting_smiles or target_smiles
        if not validate_smiles(current_smiles):
            raise ValueError(f"Invalid starting SMILES: {current_smiles!r}")

        weight_tensor = self.value_fn.weights_from_dict(objective_weights).to(self.device)
        n_steps = max_steps or self.cfg.max_steps
        history: list[SynthesisState] = []
        executed_steps: list[tuple[SynthesisState, SynthesisAction]] = []
        start_time = time.perf_counter()

        state = SynthesisState(
            target_smiles=target_smiles,
            current_smiles=current_smiles,
            inventory=inventory or [],
            reaction_history=[],
            temperature=298.15,
            pressure=1.0,
            scale=1.0,
            cost_accumulated=0.0,
            step_number=0,
            yield_so_far=1.0,
            purity_so_far=1.0,
        )

        for step in range(n_steps):
            best_action, predicted_scores = self._mpc_step(state, history, weight_tensor)

            executed_steps.append((state, best_action))
            history = (history + [state])[-self.cfg.history_len:]

            # Update state after executing action
            state = self._apply_action(state, best_action, predicted_scores)

            logger.info(
                "Step %d/%d: reaction_class=%d, cumulative_yield=%.3f",
                step + 1, n_steps, best_action.reaction_class_id, state.yield_so_far,
            )

            # Termination: check if we've reached the target
            if state.current_smiles == target_smiles:
                logger.info("Target molecule reached at step %d.", step + 1)
                break

        planning_time = time.perf_counter() - start_time
        return self._build_route(executed_steps, state, planning_time)

    def _mpc_step(
        self,
        state: SynthesisState,
        history: list[SynthesisState],
        weight_tensor: torch.Tensor,
    ) -> tuple[SynthesisAction, dict[str, float]]:
        """Execute one MPC step: sample → predict → score → select best.

        Returns:
            Tuple of (best_action, predicted_objective_scores_for_best).
        """
        state_enc = self._encode_state(state)             # (1, state_dim)
        history_enc = self._encode_history(history)        # (1, H, state_dim)

        # 1. Sample N candidate action sequences from ρ
        candidate_actions = self.action_proposal.sample(
            state_enc, history_enc, n_samples=self.cfg.n_candidates
        )  # (N, F, action_dim)

        # 2. Predict future state trajectories via p_d
        s_expanded = state_enc.expand(self.cfg.n_candidates, -1)   # (N, state_dim)
        h_expanded = history_enc.expand(self.cfg.n_candidates, -1, -1)  # (N, H, state_dim)
        predicted_trajectories = self.world_model.rollout(
            s_expanded, h_expanded, candidate_actions
        )  # (N, F, state_dim)

        # 3. Score with J (ValueFunction)
        # Build state sequences: [current_state, predicted_future_states]
        state_seq = torch.cat(
            [s_expanded.unsqueeze(1), predicted_trajectories], dim=1
        )  # (N, F+1, state_dim)

        scores = self.value_fn.score(state_seq, candidate_actions, weight_tensor)  # (N,)

        # 4. Select best candidate
        best_idx = scores.argmax().item()
        best_action_tensor = candidate_actions[best_idx, 0]  # first action of best sequence

        best_action = self._tensor_to_action(best_action_tensor)

        # Extract predicted scores for the best trajectory (for state update)
        best_predicted_state = predicted_trajectories[best_idx, 0]
        predicted_scores = {
            "yield": float(best_predicted_state[0].clamp(0, 1)),
        }

        return best_action, predicted_scores

    def _encode_state(self, state: SynthesisState) -> torch.Tensor:
        """Encode a SynthesisState to a (1, state_dim) tensor."""
        try:
            enc = self.encoder.encode([state.current_smiles], self.device)
        except Exception:
            enc = torch.zeros(1, self.encoder.hidden_dim, device=self.device)
        return enc

    def _encode_history(self, history: list[SynthesisState]) -> torch.Tensor:
        """Encode history states to (1, H, state_dim) tensor."""
        H = self.cfg.history_len
        encs: list[torch.Tensor] = []
        for s in history[-H:]:
            try:
                enc = self.encoder.encode([s.current_smiles], self.device)
            except Exception:
                enc = torch.zeros(1, self.encoder.hidden_dim, device=self.device)
            encs.append(enc)
        while len(encs) < H:
            encs.insert(0, torch.zeros(1, self.encoder.hidden_dim, device=self.device))
        return torch.stack(encs, dim=1)  # (1, H, state_dim)

    def _tensor_to_action(self, action_tensor: torch.Tensor) -> SynthesisAction:
        """Convert a (action_dim,) tensor to SynthesisAction."""
        a = action_tensor.cpu().tolist()
        return SynthesisAction(
            reaction_class_id=max(0, int(round(a[0] * 100))),
            temperature_norm=float(a[1]),
            pressure_norm=float(a[2]),
            time_norm=float(a[3]),
            solvent_id=max(0, int(round(a[4] * 64))),
            catalyst_id=max(0, int(round(a[5] * 128))),
            reagent_ratio=float(a[6]),
        )

    def _apply_action(
        self,
        state: SynthesisState,
        action: SynthesisAction,
        predicted_scores: dict[str, float],
    ) -> SynthesisState:
        """Create next state by applying action to current state."""
        return SynthesisState(
            target_smiles=state.target_smiles,
            current_smiles=state.current_smiles,   # updated by real chemistry execution
            inventory=state.inventory,
            reaction_history=state.reaction_history + [action],
            temperature=state.temperature,
            pressure=state.pressure,
            scale=state.scale,
            cost_accumulated=state.cost_accumulated,
            step_number=state.step_number + 1,
            yield_so_far=state.yield_so_far * predicted_scores.get("yield", 0.9),
            purity_so_far=state.purity_so_far,
        )

    def _build_route(
        self,
        steps: list[tuple[SynthesisState, SynthesisAction]],
        final_state: SynthesisState,
        planning_time: float,
    ) -> SynthesisRoute:
        """Assemble the final SynthesisRoute from executed steps."""
        objective_scores = {name: 0.0 for name in OBJECTIVE_NAMES}
        objective_scores["yield"] = final_state.yield_so_far
        objective_scores["purity"] = final_state.purity_so_far

        return SynthesisRoute(
            steps=steps,
            terminal_smiles=final_state.current_smiles,
            objective_scores=objective_scores,
            total_yield_fraction=final_state.yield_so_far,
            total_cost_usd=final_state.cost_accumulated,
            n_steps=len(steps),
            planning_time_seconds=planning_time,
        )


def load_models(
    checkpoint_dir: str,
    model_cfg: dict,
    device: str = "cpu",
) -> dict[str, nn.Module]:
    """Load all trained models from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing action_proposal.pt,
                        world_model.pt, value_fn.pt checkpoints.
        model_cfg: Model config dict from configs/model.yaml.
        device: Device to load models onto.

    Returns:
        Dict with keys: "action_proposal", "world_model", "value_fn", "encoder".
    """
    from pathlib import Path

    from dmpsp.action_proposal import build_action_proposal
    from dmpsp.encoder import build_encoder
    from dmpsp.utils import load_checkpoint
    from dmpsp.value_fn import build_value_fn
    from dmpsp.world_model import build_world_model

    ckpt_dir = Path(checkpoint_dir)

    encoder = build_encoder(model_cfg.get("encoder", {}))
    action_proposal = build_action_proposal(model_cfg)
    world_model = build_world_model(model_cfg)
    value_fn = build_value_fn(model_cfg)

    for name, model in [
        ("encoder", encoder),
        ("action_proposal", action_proposal),
        ("world_model", world_model),
        ("value_fn", value_fn),
    ]:
        ckpt_path = ckpt_dir / f"{name}.pt"
        if ckpt_path.exists():
            load_checkpoint(ckpt_path, model, device=device)
        else:
            logger.warning("Checkpoint not found for %s: %s", name, ckpt_path)

    return {
        "action_proposal": action_proposal,
        "world_model": world_model,
        "value_fn": value_fn,
        "encoder": encoder,
    }
