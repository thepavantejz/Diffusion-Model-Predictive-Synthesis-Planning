"""ValueFunction: predicts per-objective discounted returns for trajectory scoring.

Maps to J(s_{t:t+F}, a_{t:t+F}) from D-MPC paper (Section 3.4).

Key design decision: outputs N_OBJECTIVES separate scalar returns — never a single
weighted scalar during training. Runtime weights are applied in score() as a dot
product. This is what enables zero-retraining objective switching.

Architecture: 10-layer Transformer with one regression head per objective.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmpsp.action_proposal import TransformerBlock, _sinusoidal_embedding

logger = logging.getLogger(__name__)

# Ordered list of objective names — position i = output head i.
# Must match configs/model.yaml 'objectives' list.
OBJECTIVE_NAMES: list[str] = [
    "yield",
    "purity",
    "cost",
    "novelty",
    "fto_risk",
    "green_chem",
    "manufacturability",
    "safety",
    "robustness",
    "supply_avail",
]


class ValueFunction(nn.Module):
    """Transformer regression model predicting per-objective discounted returns.

    Takes a (states, actions) trajectory and outputs N_OBJECTIVES return values.
    Runtime weights applied externally in score() — never baked into training.

    Args:
        state_dim: Dimension of each state encoding vector.
        action_dim: Dimension of each action vector.
        token_dim: Transformer hidden size.
        n_heads: Number of attention heads.
        mlp_hidden: FFN hidden size.
        n_layers: Number of Transformer blocks.
        n_objectives: Number of output objectives.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 7,
        token_dim: int = 256,
        n_heads: int = 8,
        mlp_hidden: int = 2048,
        n_layers: int = 10,
        n_objectives: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_objectives = n_objectives

        # Project states and actions into shared token space
        self.state_proj = nn.Linear(state_dim, token_dim)
        self.action_proj = nn.Linear(action_dim, token_dim)

        # Learnable [CLS] token for trajectory-level summary
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))

        # Transformer blocks (self-attention only, no cross-attention here)
        self.blocks = nn.ModuleList([
            TransformerBlock(token_dim, n_heads, mlp_hidden, dropout, cross_attn=False)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(token_dim)

        # One regression head per objective
        self.objective_heads = nn.ModuleList([
            nn.Linear(token_dim, 1)
            for _ in range(n_objectives)
        ])

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict per-objective discounted returns for a batch of trajectories.

        Args:
            states: State sequence, shape (B, F+1, state_dim).
                    states[:, 0] is the initial state, states[:, 1:] are future states.
            actions: Action sequence, shape (B, F, action_dim).

        Returns:
            Per-objective returns, shape (B, n_objectives).
        """
        B, Fp1, _ = states.shape
        F = actions.shape[1]

        # Interleave states and actions: [s0, a0, s1, a1, ..., s_F]
        state_tokens = self.state_proj(states)     # (B, F+1, token_dim)
        action_tokens = self.action_proj(actions)   # (B, F, token_dim)

        # Build token sequence: s0, a0, s1, a1, ..., aF-1, sF
        tokens_list: list[torch.Tensor] = []
        for i in range(F):
            tokens_list.append(state_tokens[:, i:i+1, :])
            tokens_list.append(action_tokens[:, i:i+1, :])
        tokens_list.append(state_tokens[:, F:F+1, :])

        tokens = torch.cat(tokens_list, dim=1)  # (B, 2F+1, token_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 2F+2, token_dim)

        # Transformer
        for block in self.blocks:
            tokens = block(tokens)

        # Use [CLS] token output for trajectory-level prediction
        cls_out = self.out_norm(tokens[:, 0, :])  # (B, token_dim)

        # Per-objective predictions
        objective_values = torch.cat(
            [head(cls_out) for head in self.objective_heads], dim=-1
        )  # (B, n_objectives)

        return objective_values

    def training_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_returns: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss between predicted and target per-objective returns.

        Args:
            states: State sequence, shape (B, F+1, state_dim).
            actions: Action sequence, shape (B, F, action_dim).
            target_returns: Ground-truth returns, shape (B, n_objectives).

        Returns:
            Scalar MSE loss.
        """
        predicted = self.forward(states, actions)
        return F.mse_loss(predicted, target_returns)

    def score(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Score trajectories using runtime objective weights.

        This is the runtime inference path. Weights are applied as a dot product
        on top of the per-objective predictions. No retraining needed when weights change.

        Args:
            states: State sequence, shape (B, F+1, state_dim).
            actions: Action sequence, shape (B, F, action_dim).
            weights: Objective weights, shape (n_objectives,). Need not sum to 1.

        Returns:
            Scalar score per trajectory, shape (B,).
        """
        objective_values = self.forward(states, actions)   # (B, n_objectives)
        w = weights.to(objective_values.device)
        return (objective_values * w.unsqueeze(0)).sum(dim=-1)  # (B,)

    def weights_from_dict(self, weight_dict: dict[str, float]) -> torch.Tensor:
        """Convert a dict of {objective_name: weight} to a tensor.

        Missing objectives default to 0.0.

        Args:
            weight_dict: E.g. {"yield": 0.4, "cost": 0.3, "safety": 0.3}

        Returns:
            Tensor of shape (n_objectives,).
        """
        weights = torch.zeros(self.n_objectives)
        for i, name in enumerate(OBJECTIVE_NAMES[:self.n_objectives]):
            weights[i] = float(weight_dict.get(name, 0.0))
        return weights


def build_value_fn(cfg: dict) -> ValueFunction:
    """Construct ValueFunction from config dict.

    Args:
        cfg: Full model config dict (from configs/model.yaml).

    Returns:
        ValueFunction instance.
    """
    vf_cfg = cfg.get("value_fn", {})
    enc_cfg = cfg.get("encoder", {})
    return ValueFunction(
        state_dim=enc_cfg.get("hidden_dim", 256),
        action_dim=7,
        token_dim=vf_cfg.get("token_dim", 256),
        n_heads=vf_cfg.get("n_heads", 8),
        mlp_hidden=vf_cfg.get("mlp_hidden", 2048),
        n_layers=vf_cfg.get("n_layers", 10),
        n_objectives=vf_cfg.get("n_objectives", 10),
        dropout=vf_cfg.get("dropout", 0.1),
    )
