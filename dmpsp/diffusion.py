"""Diffusion utilities shared by ActionProposalDiffusion and DynamicsDiffusion.

Implements:
  - Cosine noise schedule (Nichol & Dhariwal 2021)
  - Forward diffusion process q_sample
  - DDPM reverse step (for ActionProposalDiffusion, stochastic)
  - DDIM reverse step (for DynamicsDiffusion, deterministic by default)
  - Diffusion loss (MSE on predicted noise)

All functions are pure (no nn.Module state). They work with arbitrary batch shapes:
the last dimension is always the feature dimension, all leading dims are batch/time dims.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule from Nichol & Dhariwal (2021), Improved DDPM.

    Args:
        timesteps: Total diffusion steps T.
        s: Small offset preventing singularity near t=0.

    Returns:
        Tensor of shape (timesteps,) with beta values in (0, 1).
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, min=1e-5, max=0.9999).float()


def make_schedule(timesteps: int, s: float = 0.008) -> dict[str, torch.Tensor]:
    """Precompute all schedule tensors for a given number of diffusion steps.

    Args:
        timesteps: Total diffusion steps T.
        s: Cosine schedule offset.

    Returns:
        Dict with keys:
            betas, alphas, alphas_cumprod, alphas_cumprod_prev,
            sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
            posterior_variance.
    """
    betas = cosine_beta_schedule(timesteps, s)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": alphas_cumprod.sqrt(),
        "sqrt_one_minus_alphas_cumprod": (1.0 - alphas_cumprod).sqrt(),
        "posterior_variance": (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        ),
    }


def q_sample(
    x_start: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: add noise to x_start at timestep t.

    x_t = sqrt(alpha_bar_t) * x_start + sqrt(1 - alpha_bar_t) * noise

    Args:
        x_start: Clean data, shape (B, ..., D).
        t: Integer timesteps, shape (B,). Values in [0, T-1].
        sqrt_alphas_cumprod: Precomputed schedule, shape (T,).
        sqrt_one_minus_alphas_cumprod: Precomputed schedule, shape (T,).
        noise: Optional pre-sampled noise (same shape as x_start).

    Returns:
        Tuple (x_t, noise): noisy sample and the noise that was added.
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    sa = sqrt_alphas_cumprod[t]
    soma = sqrt_one_minus_alphas_cumprod[t]

    # Expand scalar schedule values to broadcast against x_start's trailing dims
    while sa.dim() < x_start.dim():
        sa = sa.unsqueeze(-1)
        soma = soma.unsqueeze(-1)

    return sa * x_start + soma * noise, noise


def compute_diffusion_loss(
    predicted_noise: torch.Tensor,
    actual_noise: torch.Tensor,
) -> torch.Tensor:
    """Simple MSE loss on predicted vs actual noise.

    Args:
        predicted_noise: Model output (denoised noise prediction).
        actual_noise: Ground-truth noise from q_sample.

    Returns:
        Scalar mean MSE loss.
    """
    return F.mse_loss(predicted_noise, actual_noise)


def ddpm_reverse_step(
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Single DDPM reverse step: x_t → x_{t-1}.

    Used by ActionProposalDiffusion (stochastic sampling).

    Args:
        model_output: Predicted noise from denoising model, shape (B, ..., D).
        x_t: Noisy sample at timestep t, same shape.
        t: Current integer timestep (scalar).
        schedule: Output of make_schedule().

    Returns:
        x_{t-1} sample, same shape as x_t.
    """
    beta_t = schedule["betas"][t]
    soma_t = schedule["sqrt_one_minus_alphas_cumprod"][t]
    recip_sqrt_alpha_t = (1.0 / schedule["alphas"][t]).sqrt()
    posterior_var_t = schedule["posterior_variance"][t]

    def _b(v: torch.Tensor) -> torch.Tensor:
        while v.dim() < x_t.dim():
            v = v.unsqueeze(-1)
        return v

    # Predicted mean of p(x_{t-1} | x_t)
    mean = _b(recip_sqrt_alpha_t) * (
        x_t - _b(beta_t) / _b(soma_t) * model_output
    )

    if t == 0:
        return mean

    return mean + _b(posterior_var_t).sqrt() * torch.randn_like(x_t)


def ddim_reverse_step(
    model_output: torch.Tensor,
    x_t: torch.Tensor,
    t: int,
    t_prev: int,
    schedule: dict[str, torch.Tensor],
    eta: float = 0.0,
) -> torch.Tensor:
    """Single DDIM reverse step: x_t → x_{t_prev}.

    Used by DynamicsDiffusion (deterministic by default, eta=0.0).

    Args:
        model_output: Predicted noise from denoising model, shape (B, ..., D).
        x_t: Noisy sample at timestep t, same shape.
        t: Current integer timestep.
        t_prev: Target timestep (t_prev < t).
        schedule: Output of make_schedule().
        eta: Stochasticity coefficient. 0.0 = fully deterministic DDIM.

    Returns:
        x_{t_prev} sample, same shape as x_t.
    """
    acp = schedule["alphas_cumprod"]
    alpha_t = acp[t]
    alpha_prev = acp[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)

    def _b(v: torch.Tensor) -> torch.Tensor:
        v = v.to(x_t.device)
        while v.dim() < x_t.dim():
            v = v.unsqueeze(-1)
        return v

    alpha_t = _b(alpha_t)
    alpha_prev = _b(alpha_prev)

    # Predict x_0 from x_t and noise estimate
    x0_pred = (x_t - (1.0 - alpha_t).sqrt() * model_output) / alpha_t.sqrt()

    # Variance of the backward step
    sigma_t = eta * (
        (1.0 - alpha_prev) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_prev)
    ).clamp(min=0.0).sqrt()

    # Direction pointing toward x_t
    dir_xt = (1.0 - alpha_prev - sigma_t ** 2).clamp(min=0.0).sqrt() * model_output

    noise = sigma_t * torch.randn_like(x_t) if eta > 0.0 else 0.0
    return alpha_prev.sqrt() * x0_pred + dir_xt + noise


class EMAModel:
    """Exponential moving average of model weights.

    Maintains a shadow copy of model parameters. Call update() after each
    optimizer step. Use ema_model.state_dict() to save or evaluate.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.99) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}.")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.clone().float() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA weights from current model weights."""
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k] = self.decay * self.shadow[k] + (1.0 - self.decay) * v.float()

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return the EMA weight dict (compatible with model.load_state_dict)."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def copy_to(self, model: torch.nn.Module) -> None:
        """Copy EMA weights into model for evaluation."""
        model.load_state_dict(self.state_dict())
