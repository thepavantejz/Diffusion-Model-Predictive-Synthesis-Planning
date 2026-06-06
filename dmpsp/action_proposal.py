"""ActionProposalDiffusion: learns a distribution over reaction sequences.

Maps to ρ(a_{t:t+F} | s_t, h_t) from D-MPC paper (Section 3.2).

Architecture:
  - Transformer encoder with cross-attention conditioning
  - DDPM sampling (stochastic, 32 steps by default)
  - Trains via denoising score matching on offline reaction data
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmpsp.diffusion import (
    EMAModel,
    compute_diffusion_loss,
    ddim_reverse_step,
    ddpm_reverse_step,
    make_schedule,
    q_sample,
)

logger = logging.getLogger(__name__)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional cross-attention.

    Used in both ActionProposalDiffusion and DynamicsDiffusion.
    """

    def __init__(
        self,
        token_dim: int,
        n_heads: int,
        mlp_hidden: int,
        dropout: float = 0.1,
        cross_attn: bool = False,
        context_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(token_dim)
        self.self_attn = nn.MultiheadAttention(
            token_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(token_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, token_dim),
            nn.Dropout(dropout),
        )
        self.cross_attn = None
        if cross_attn:
            ctx_dim = context_dim or token_dim
            self.norm_cross = nn.LayerNorm(token_dim)
            self.cross_attn = nn.MultiheadAttention(
                token_dim, n_heads, dropout=dropout, batch_first=True,
                kdim=ctx_dim, vdim=ctx_dim,
            )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # Cross-attention (conditioning on state + history context)
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm_cross(x), context, context)[0]
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


def _sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal diffusion timestep embedding.

    Args:
        t: Integer timesteps, shape (B,).
        dim: Embedding dimension.

    Returns:
        Tensor of shape (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float, device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ActionProposalDiffusion(nn.Module):
    """Diffusion model that proposes F-step reaction sequences.

    Conditioning: current state encoding + history encoding (cross-attention).
    Output distribution: over (F, action_dim) action sequences.
    Sampling: DDPM (stochastic).

    Args:
        action_dim: Dimension of each action vector.
        state_dim: Dimension of state encoding from encoder.
        token_dim: Internal Transformer hidden size.
        n_heads: Number of attention heads.
        mlp_hidden: Transformer FFN hidden size.
        n_layers: Number of Transformer blocks.
        n_diffusion_steps: T for DDPM schedule.
        horizon: F — planning horizon.
        history_len: H — history context length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        action_dim: int = 7,
        state_dim: int = 256,
        token_dim: int = 256,
        n_heads: int = 8,
        mlp_hidden: int = 2048,
        n_layers: int = 5,
        n_diffusion_steps: int = 32,
        horizon: int = 10,
        history_len: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.token_dim = token_dim
        self.horizon = horizon
        self.history_len = history_len
        self.n_diffusion_steps = n_diffusion_steps

        # Precompute diffusion schedule (registered as buffers so they move with .to(device))
        schedule = make_schedule(n_diffusion_steps)
        for k, v in schedule.items():
            self.register_buffer(k, v)

        # Project noisy action sequence to token_dim
        self.action_proj = nn.Linear(action_dim, token_dim)

        # Diffusion timestep embedding
        self.time_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.SiLU(),
            nn.Linear(token_dim * 2, token_dim),
        )

        # Positional encoding for the F-step sequence
        self.pos_emb = nn.Embedding(horizon, token_dim)

        # Context projection: (state + H*state) → context tokens for cross-attn
        context_in = state_dim * (1 + history_len)
        self.context_proj = nn.Linear(context_in, token_dim)

        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlock(
                token_dim, n_heads, mlp_hidden, dropout,
                cross_attn=True, context_dim=token_dim,
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Linear(token_dim, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        diffusion_t: torch.Tensor,
        state_enc: torch.Tensor,
        history_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise component of noisy_actions.

        Args:
            noisy_actions: Noisy action sequence, shape (B, F, action_dim).
            diffusion_t: Integer diffusion timesteps, shape (B,).
            state_enc: Current state encoding, shape (B, state_dim).
            history_enc: History state encodings, shape (B, H, state_dim).

        Returns:
            Predicted noise, same shape as noisy_actions: (B, F, action_dim).
        """
        B, F, _ = noisy_actions.shape

        # Project actions to token space
        x = self.action_proj(noisy_actions)  # (B, F, token_dim)

        # Add positional embeddings
        positions = torch.arange(F, device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)

        # Add timestep embedding (broadcast across F)
        t_emb = self.time_proj(_sinusoidal_embedding(diffusion_t, self.token_dim))
        x = x + t_emb.unsqueeze(1)

        # Build context tensor from state + history
        history_flat = history_enc.reshape(B, -1)           # (B, H*state_dim)
        context_flat = torch.cat([state_enc, history_flat], dim=-1)  # (B, (1+H)*state_dim)
        context = self.context_proj(context_flat).unsqueeze(1)   # (B, 1, token_dim)

        # Transformer blocks with cross-attention to context
        for block in self.blocks:
            x = block(x, context=context)

        x = self.out_norm(x)
        return self.out_proj(x)  # (B, F, action_dim) — predicted noise

    def training_loss(
        self,
        actions: torch.Tensor,
        state_enc: torch.Tensor,
        history_enc: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DDPM denoising loss.

        Args:
            actions: Ground-truth action sequences, shape (B, F, action_dim).
            state_enc: Current state encoding, shape (B, state_dim).
            history_enc: History state encodings, shape (B, H, state_dim).

        Returns:
            Scalar MSE loss.
        """
        B = actions.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=actions.device)
        x_t, noise = q_sample(
            actions, t,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        )
        predicted_noise = self.forward(x_t, t, state_enc, history_enc)
        return compute_diffusion_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        state_enc: torch.Tensor,
        history_enc: torch.Tensor,
        n_samples: int = 64,
        deterministic: bool = True,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample N candidate action sequences.

        Args:
            state_enc: State encoding, shape (1, state_dim) or (B, state_dim).
            history_enc: History encoding, shape (1, H, state_dim) or (B, H, state_dim).
            n_samples: Number of candidate sequences to sample.
            deterministic: If True, use DDIM (eta=0) — fully reproducible given seed.
                           If False, use DDPM (stochastic, richer diversity).
            seed: Optional RNG seed. When set, starting noise is fixed → identical
                  outputs across calls with the same inputs.

        Returns:
            Sampled action sequences, shape (n_samples, F, action_dim).
        """
        device = state_enc.device
        s = state_enc.expand(n_samples, -1)
        h = history_enc.expand(n_samples, -1, -1)

        # Fix starting noise if seed given
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            x = torch.randn(
                n_samples, self.horizon, self.action_dim, device=device, generator=gen
            )
        else:
            x = torch.randn(n_samples, self.horizon, self.action_dim, device=device)

        sched = self._schedule_dict()

        if deterministic:
            # DDIM reverse process (eta=0): deterministic denoising
            for t_int in reversed(range(self.n_diffusion_steps)):
                t_batch = torch.full((n_samples,), t_int, dtype=torch.long, device=device)
                predicted_noise = self.forward(x, t_batch, s, h)
                x = ddim_reverse_step(predicted_noise, x, t_int, t_int - 1, sched, eta=0.0)
        else:
            # DDPM reverse process (stochastic)
            for t_int in reversed(range(self.n_diffusion_steps)):
                t_batch = torch.full((n_samples,), t_int, dtype=torch.long, device=device)
                predicted_noise = self.forward(x, t_batch, s, h)
                x = ddpm_reverse_step(predicted_noise, x, t_int, sched)

        return x

    def _schedule_dict(self) -> dict[str, torch.Tensor]:
        return {
            "betas": self.betas,
            "alphas": self.alphas,
            "alphas_cumprod": self.alphas_cumprod,
            "sqrt_alphas_cumprod": self.sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": self.sqrt_one_minus_alphas_cumprod,
            "posterior_variance": self.posterior_variance,
        }


def build_action_proposal(cfg: dict) -> ActionProposalDiffusion:
    """Construct ActionProposalDiffusion from config dict.

    Args:
        cfg: Dict with action_proposal sub-dict from configs/model.yaml.

    Returns:
        ActionProposalDiffusion instance.
    """
    ap_cfg = cfg.get("action_proposal", {})
    enc_cfg = cfg.get("encoder", {})
    return ActionProposalDiffusion(
        action_dim=7,
        state_dim=enc_cfg.get("hidden_dim", 256),
        token_dim=ap_cfg.get("token_dim", 256),
        n_heads=ap_cfg.get("n_heads", 8),
        mlp_hidden=ap_cfg.get("mlp_hidden", 2048),
        n_layers=ap_cfg.get("n_layers", 5),
        n_diffusion_steps=ap_cfg.get("n_diffusion_steps", 32),
        horizon=ap_cfg.get("horizon", 10),
        history_len=ap_cfg.get("history_len", 1),
        dropout=ap_cfg.get("dropout", 0.1),
    )
