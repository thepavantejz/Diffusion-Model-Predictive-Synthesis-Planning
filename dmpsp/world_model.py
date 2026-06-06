"""ChemistryWorldModel: predicts multi-step synthesis outcomes.

Maps to p_d(s_{t+1:t+F} | s_t, h_t, a_{t:t+F}) from D-MPC paper (Section 3.3).

Architecture:
  - ReactionT5 backbone (sagawa/ReactionT5) fine-tuned on task data
  - 5 property prediction heads (yield, toxicity, manufacturability, supply chain, patentability)
  - DynamicsDiffusion wrapper for joint F-step prediction via DDIM

The backbone and heads are trained together (freeze_backbone=False by default).
For dynamics-only fine-tuning (new catalyst/conditions), set freeze_backbone=True
and only retrain the property heads and diffusion wrapper — this is the D-MPC
factored fine-tuning strategy.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmpsp.action_proposal import TransformerBlock, _sinusoidal_embedding
from dmpsp.diffusion import (
    compute_diffusion_loss,
    ddim_reverse_step,
    make_schedule,
    q_sample,
)

logger = logging.getLogger(__name__)

# Backbone model ID on HuggingFace
_REACTIONT5_MODEL_ID = "sagawa/ReactionT5"


class PropertyHeads(nn.Module):
    """Lightweight MLP heads that predict synthesis properties from backbone features.

    All heads take encoder hidden states from ReactionT5 and predict scalar scores.
    Outputs are in [0, 1] (sigmoid-activated).
    """

    def __init__(self, backbone_hidden: int, hidden: int = 256) -> None:
        super().__init__()

        def _head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(backbone_hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
                nn.Sigmoid(),
            )

        self.yield_head = _head()
        self.purity_head = _head()
        self.toxicity_head = _head()
        self.manufacturability_head = _head()
        self.supply_chain_head = _head()
        self.patentability_head = _head()

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict all property scores from backbone hidden states.

        Args:
            hidden: Mean-pooled backbone hidden states, shape (B, backbone_hidden).

        Returns:
            Dict mapping property name to scalar tensor of shape (B, 1).
        """
        return {
            "yield": self.yield_head(hidden),
            "purity": self.purity_head(hidden),
            "toxicity": self.toxicity_head(hidden),
            "manufacturability": self.manufacturability_head(hidden),
            "supply_avail": self.supply_chain_head(hidden),
            "patentability": self.patentability_head(hidden),
        }


class ChemistryWorldModel(nn.Module):
    """Full chemistry world model combining ReactionT5 + property heads + DynamicsDiffusion.

    The model does two things:
    1. predict_properties(): given a reaction SMILES string, return property scores
    2. rollout(): given a state + action sequence, predict F future state vectors via DDIM

    Training:
    - Phase 1: train_world_model.py fine-tunes backbone + heads on reaction data
    - Phase 2: rollout (DynamicsDiffusion) is trained separately on trajectory data

    Args:
        backbone_id: HuggingFace model ID for ReactionT5.
        freeze_backbone: If True, backbone weights are frozen (inference or fine-tune only heads).
        gradient_checkpointing: Enable for 16GB GPU (Kaggle P100).
        state_dim: Dimension of state encoding vectors.
        action_dim: Dimension of action vectors.
        token_dim: Internal Transformer hidden size for DynamicsDiffusion.
        n_heads: Attention heads.
        mlp_hidden: FFN hidden size.
        n_layers: Transformer layers for DynamicsDiffusion.
        n_diffusion_steps: DDIM steps.
        eta: DDIM stochasticity (0.0 = deterministic).
        horizon: F — planning horizon.
        history_len: H — history context length.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        backbone_id: str = _REACTIONT5_MODEL_ID,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
        state_dim: int = 256,
        action_dim: int = 7,
        token_dim: int = 256,
        n_heads: int = 8,
        mlp_hidden: int = 2048,
        n_layers: int = 5,
        n_diffusion_steps: int = 10,
        eta: float = 0.0,
        horizon: int = 10,
        history_len: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.eta = eta
        self.horizon = horizon
        self.history_len = history_len
        self.n_diffusion_steps = n_diffusion_steps
        self.state_dim = state_dim

        # Load ReactionT5 backbone
        self.backbone, self.tokenizer, backbone_hidden = _load_reactiont5(
            backbone_id, freeze_backbone, gradient_checkpointing
        )

        # Property prediction heads
        self.property_heads = PropertyHeads(backbone_hidden)

        # DynamicsDiffusion Transformer (operates on state space, conditioned on actions)
        self.state_proj = nn.Linear(state_dim, token_dim)
        self.action_proj = nn.Linear(action_dim, token_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.SiLU(),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.pos_emb = nn.Embedding(horizon, token_dim)
        context_in = state_dim * (1 + history_len) + action_dim * horizon
        self.context_proj = nn.Linear(context_in, token_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                token_dim, n_heads, mlp_hidden, dropout,
                cross_attn=True, context_dim=token_dim,
            )
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(token_dim)
        self.out_proj = nn.Linear(token_dim, state_dim)

        # DDIM schedule
        schedule = make_schedule(n_diffusion_steps)
        for k, v in schedule.items():
            self.register_buffer(k, v)

    def predict_properties(
        self, reaction_smiles: list[str], device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Predict synthesis properties for a batch of reaction SMILES strings.

        Args:
            reaction_smiles: List of reaction SMILES in format "reactants>>product".
            device: Target device.

        Returns:
            Dict mapping property name → Tensor of shape (B, 1), values in [0, 1].
        """
        tokens = self.tokenizer(
            reaction_smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        outputs = self.backbone(**tokens)
        # Mean-pool encoder last hidden states
        hidden = outputs.encoder_last_hidden_state.mean(dim=1)  # (B, backbone_hidden)
        return self.property_heads(hidden)

    def _dynamics_forward(
        self,
        noisy_states: torch.Tensor,
        diffusion_t: torch.Tensor,
        state_enc: torch.Tensor,
        history_enc: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """DynamicsDiffusion denoiser forward pass.

        Args:
            noisy_states: Noisy future states, shape (B, F, state_dim).
            diffusion_t: Diffusion timesteps, shape (B,).
            state_enc: Current state, shape (B, state_dim).
            history_enc: History states, shape (B, H, state_dim).
            actions: Proposed action sequence, shape (B, F, action_dim).

        Returns:
            Predicted noise, shape (B, F, state_dim).
        """
        B, F, _ = noisy_states.shape

        x = self.state_proj(noisy_states)
        positions = torch.arange(F, device=x.device)
        x = x + self.pos_emb(positions).unsqueeze(0)
        t_emb = self.time_proj(_sinusoidal_embedding(diffusion_t, x.shape[-1]))
        x = x + t_emb.unsqueeze(1)

        # Context: current state + history + actions (all concatenated)
        history_flat = history_enc.reshape(B, -1)
        actions_flat = actions.reshape(B, -1)
        context_flat = torch.cat([state_enc, history_flat, actions_flat], dim=-1)
        context = self.context_proj(context_flat).unsqueeze(1)  # (B, 1, token_dim)

        for block in self.blocks:
            x = block(x, context=context)

        return self.out_proj(self.out_norm(x))

    def dynamics_loss(
        self,
        future_states: torch.Tensor,
        state_enc: torch.Tensor,
        history_enc: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DDIM denoising loss for the dynamics model.

        Args:
            future_states: Ground-truth future state encodings, shape (B, F, state_dim).
            state_enc: Current state encoding, shape (B, state_dim).
            history_enc: History state encodings, shape (B, H, state_dim).
            actions: Action sequence, shape (B, F, action_dim).

        Returns:
            Scalar MSE loss.
        """
        B = future_states.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (B,), device=future_states.device)
        x_t, noise = q_sample(
            future_states, t,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
        )
        predicted_noise = self._dynamics_forward(x_t, t, state_enc, history_enc, actions)
        return compute_diffusion_loss(predicted_noise, noise)

    @torch.no_grad()
    def rollout(
        self,
        state_enc: torch.Tensor,
        history_enc: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict F future state encodings via DDIM (deterministic by default).

        Args:
            state_enc: Current state encoding, shape (B, state_dim).
            history_enc: History state encodings, shape (B, H, state_dim).
            actions: Proposed action sequence, shape (B, F, action_dim).

        Returns:
            Predicted future state sequence, shape (B, F, state_dim).
        """
        B = state_enc.shape[0]
        F = actions.shape[1]
        device = state_enc.device

        x = torch.randn(B, F, self.state_dim, device=device)

        # Build DDIM timestep sequence (evenly spaced subset of [0, T])
        timesteps = list(range(0, self.n_diffusion_steps))[::-1]

        for i, t_int in enumerate(timesteps):
            t_batch = torch.full((B,), t_int, dtype=torch.long, device=device)
            predicted_noise = self._dynamics_forward(x, t_batch, state_enc, history_enc, actions)
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = ddim_reverse_step(
                predicted_noise, x, t_int, t_prev, self._schedule_dict(), self.eta
            )

        return x

    def _schedule_dict(self) -> dict[str, torch.Tensor]:
        return {
            "betas": self.betas,
            "alphas": self.alphas,
            "alphas_cumprod": self.alphas_cumprod,
            "alphas_cumprod_prev": self.alphas_cumprod_prev,
            "sqrt_alphas_cumprod": self.sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": self.sqrt_one_minus_alphas_cumprod,
            "posterior_variance": self.posterior_variance,
        }


def _load_reactiont5(
    model_id: str,
    freeze: bool,
    gradient_checkpointing: bool,
) -> tuple[nn.Module, object, int]:
    """Load ReactionT5 model and tokenizer from HuggingFace.

    Returns:
        Tuple of (model, tokenizer, hidden_size).

    Raises:
        ImportError: If transformers is not installed.
        RuntimeError: If model download fails.
    """
    try:
        from transformers import AutoTokenizer, T5ForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "ChemistryWorldModel requires 'transformers'. "
            "Install with: pip install transformers"
        ) from exc

    logger.info("Loading ReactionT5 backbone: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(model_id)

    if gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        logger.info("ReactionT5 backbone frozen (freeze_backbone=True).")

    hidden_size: int = model.config.d_model
    logger.info("ReactionT5 loaded: d_model=%d, freeze=%s", hidden_size, freeze)
    return model, tokenizer, hidden_size


def build_world_model(cfg: dict) -> ChemistryWorldModel:
    """Construct ChemistryWorldModel from config dict.

    Args:
        cfg: Full model config dict (from configs/model.yaml).

    Returns:
        ChemistryWorldModel instance.
    """
    wm_cfg = cfg.get("world_model", {})
    enc_cfg = cfg.get("encoder", {})
    return ChemistryWorldModel(
        backbone_id=wm_cfg.get("backbone_id", _REACTIONT5_MODEL_ID),
        freeze_backbone=wm_cfg.get("freeze_backbone", False),
        gradient_checkpointing=wm_cfg.get("gradient_checkpointing", True),
        state_dim=enc_cfg.get("hidden_dim", 256),
        action_dim=7,
        token_dim=wm_cfg.get("token_dim", 256),
        n_heads=wm_cfg.get("n_heads", 8),
        mlp_hidden=wm_cfg.get("mlp_hidden", 2048),
        n_layers=wm_cfg.get("n_layers", 5),
        n_diffusion_steps=wm_cfg.get("n_diffusion_steps", 10),
        eta=wm_cfg.get("eta", 0.0),
        horizon=cfg.get("action_proposal", {}).get("horizon", 10),
        history_len=cfg.get("action_proposal", {}).get("history_len", 1),
        dropout=wm_cfg.get("dropout", 0.1),
    )
