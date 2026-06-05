"""PyTorch Dataset for DMPSP training.

SynthesisDataset wraps a list of DMPSPTrajectory and serves batches
in three modes matching the three trainable components:

  proposal  — (state_tensor, history_tensor, action_sequence)
  dynamics  — (state_tensor, history_tensor, action_sequence, future_states)
  value     — (state_sequence, action_sequence, return_vector)

All tensors are computed lazily from the stored dataclass objects.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch.utils.data import Dataset

from dmpsp.state import DMPSPTrajectory, SynthesisAction, SynthesisState

logger = logging.getLogger(__name__)

Mode = Literal["proposal", "dynamics", "value"]


class SynthesisDataset(Dataset):
    """Dataset serving training samples from a list of DMPSPTrajectory.

    Args:
        trajectories: List of DMPSPTrajectory objects.
        mode: Training mode — "proposal", "dynamics", or "value".
        encoder_fn: Callable(smiles: str) → Tensor(hidden_dim). Called lazily
                    to encode molecule SMILES to embedding vectors.
                    If None, a zero vector of size encoder_hidden_dim is used.
        encoder_hidden_dim: Size of molecule embedding vectors.
        horizon: F — number of steps to include in each sample.
        history_len: H — number of past states to include as context.
        n_objectives: Number of reward objectives (must match model config).
        action_dim: Dimension of action tensor (must match model config).

    Raises:
        ValueError: If mode is not one of the three valid modes.
    """

    _VALID_MODES: frozenset[str] = frozenset({"proposal", "dynamics", "value"})

    def __init__(
        self,
        trajectories: list[DMPSPTrajectory],
        mode: Mode,
        encoder_fn=None,
        encoder_hidden_dim: int = 256,
        horizon: int = 10,
        history_len: int = 1,
        n_objectives: int = 10,
        action_dim: int = 7,    # reaction_class_id + 6 continuous fields
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"Invalid mode: {mode!r}. Choose from: {sorted(self._VALID_MODES)}"
            )
        self.trajectories = trajectories
        self.mode = mode
        self.encoder_fn = encoder_fn
        self.encoder_hidden_dim = encoder_hidden_dim
        self.horizon = horizon
        self.history_len = history_len
        self.n_objectives = n_objectives
        self.action_dim = action_dim

        # Pre-index all valid (trajectory, step) pairs
        self._index: list[tuple[int, int]] = []
        for traj_idx, traj in enumerate(trajectories):
            for step in range(len(traj)):
                self._index.append((traj_idx, step))

        logger.info(
            "SynthesisDataset[%s]: %d trajectories → %d samples",
            mode, len(trajectories), len(self._index),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx, step = self._index[idx]
        traj = self.trajectories[traj_idx]

        if self.mode == "proposal":
            return self._get_proposal_sample(traj, step)
        if self.mode == "dynamics":
            return self._get_dynamics_sample(traj, step)
        return self._get_value_sample(traj, step)

    # ------------------------------------------------------------------
    # Mode-specific sample builders
    # ------------------------------------------------------------------

    def _get_proposal_sample(self, traj: DMPSPTrajectory, step: int) -> dict[str, torch.Tensor]:
        """Sample for ActionProposalDiffusion training.

        Returns:
            state_enc: (hidden_dim,) — current state encoding
            history_enc: (history_len, hidden_dim) — past states
            actions: (F, action_dim) — ground-truth action sequence
        """
        state = traj.states[step]
        F = min(self.horizon, len(traj) - step)

        return {
            "state_enc": self._encode_state(state),
            "history_enc": self._encode_history(traj, step),
            "actions": self._actions_to_tensor(traj.actions[step:step + F], F),
        }

    def _get_dynamics_sample(self, traj: DMPSPTrajectory, step: int) -> dict[str, torch.Tensor]:
        """Sample for DynamicsDiffusion training.

        Returns:
            state_enc: (hidden_dim,)
            history_enc: (history_len, hidden_dim)
            actions: (F, action_dim)
            future_states: (F, hidden_dim) — prediction targets
        """
        state = traj.states[step]
        F = min(self.horizon, len(traj) - step)

        future_states = torch.stack([
            self._encode_state(traj.states[step + i + 1])
            for i in range(F)
        ])
        # Pad to horizon if shorter
        if F < self.horizon:
            pad = torch.zeros(self.horizon - F, self.encoder_hidden_dim)
            future_states = torch.cat([future_states, pad], dim=0)

        return {
            "state_enc": self._encode_state(state),
            "history_enc": self._encode_history(traj, step),
            "actions": self._actions_to_tensor(traj.actions[step:step + F], self.horizon),
            "future_states": future_states,
        }

    def _get_value_sample(self, traj: DMPSPTrajectory, step: int) -> dict[str, torch.Tensor]:
        """Sample for ValueFunction training.

        Returns:
            state_sequence: (F+1, hidden_dim) — states from step to step+F
            actions: (F, action_dim)
            returns: (n_objectives,) — discounted per-objective returns
        """
        F = min(self.horizon, len(traj) - step)

        state_seq = torch.stack([
            self._encode_state(traj.states[step + i])
            for i in range(F + 1)
        ])
        if F + 1 < self.horizon + 1:
            pad = torch.zeros(self.horizon + 1 - (F + 1), self.encoder_hidden_dim)
            state_seq = torch.cat([state_seq, pad], dim=0)

        returns = self._compute_discounted_returns(traj, step, F)

        return {
            "state_sequence": state_seq,
            "actions": self._actions_to_tensor(traj.actions[step:step + F], self.horizon),
            "returns": returns,
        }

    # ------------------------------------------------------------------
    # Tensor conversion helpers
    # ------------------------------------------------------------------

    def _encode_state(self, state: SynthesisState) -> torch.Tensor:
        """Encode a SynthesisState to a fixed-size tensor."""
        if self.encoder_fn is not None:
            try:
                mol_enc = self.encoder_fn(state.current_smiles)
            except Exception:
                mol_enc = torch.zeros(self.encoder_hidden_dim)
        else:
            mol_enc = torch.zeros(self.encoder_hidden_dim)

        # Append scalar context features (7 values, normalized to [0,1])
        scalars = torch.tensor([
            state.yield_so_far,
            state.purity_so_far,
            min(state.cost_accumulated / 1000.0, 1.0),
            min(state.step_number / 20.0, 1.0),
            (state.temperature - 200.0) / 250.0,    # rough normalization
            (state.pressure - 0.1) / 20.0,
            min(state.scale / 1000.0, 1.0),
        ], dtype=torch.float)

        # Zero-pad scalars to encoder_hidden_dim so concat gives 2 × hidden_dim
        # (This design matches StateEncoder in the model layer)
        return mol_enc  # the model's StateEncoder handles the full concat

    def _encode_history(self, traj: DMPSPTrajectory, step: int) -> torch.Tensor:
        """Encode the H most recent states as context."""
        H = self.history_len
        history_states = traj.states[max(0, step - H):step]
        encs = [self._encode_state(s) for s in history_states]
        # Pad with zeros if fewer than H states available
        while len(encs) < H:
            encs.insert(0, torch.zeros(self.encoder_hidden_dim))
        return torch.stack(encs)  # (H, hidden_dim)

    def _action_to_tensor(self, action: SynthesisAction) -> torch.Tensor:
        """Convert a SynthesisAction to a flat tensor of shape (action_dim,)."""
        return torch.tensor([
            float(action.reaction_class_id) / 100.0,   # normalize by num_classes
            action.temperature_norm,
            action.pressure_norm,
            action.time_norm,
            float(action.solvent_id) / 64.0,            # normalize by num_solvents
            float(action.catalyst_id) / 128.0,          # normalize by num_catalysts
            action.reagent_ratio,
        ], dtype=torch.float)

    def _actions_to_tensor(
        self, actions: list[SynthesisAction], pad_to: int
    ) -> torch.Tensor:
        """Convert a list of actions to a (pad_to, action_dim) tensor with zero padding."""
        tensors = [self._action_to_tensor(a) for a in actions]
        while len(tensors) < pad_to:
            tensors.append(torch.zeros(self.action_dim))
        return torch.stack(tensors[:pad_to])

    def _compute_discounted_returns(
        self, traj: DMPSPTrajectory, step: int, F: int
    ) -> torch.Tensor:
        """Compute per-objective discounted returns from step over F steps.

        Returns:
            Tensor of shape (n_objectives,).
        """
        gamma = 0.99
        returns = torch.zeros(self.n_objectives)
        for i in range(F):
            t = step + i
            if t < len(traj.rewards_per_objective):
                r = torch.tensor(traj.rewards_per_objective[t], dtype=torch.float)
                returns += (gamma ** i) * r
        return returns
