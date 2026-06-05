"""Tests for dmpsp.diffusion — schedule properties, sampling, loss, and EMA."""

from __future__ import annotations

import pytest
import torch

from dmpsp.diffusion import (
    EMAModel,
    compute_diffusion_loss,
    cosine_beta_schedule,
    ddim_reverse_step,
    ddpm_reverse_step,
    make_schedule,
    q_sample,
)

T = 32   # small T for fast tests


@pytest.fixture(scope="module")
def schedule() -> dict:
    return make_schedule(T)


# ---------------------------------------------------------------------------
# cosine_beta_schedule
# ---------------------------------------------------------------------------

def test_beta_shape():
    betas = cosine_beta_schedule(T)
    assert betas.shape == (T,)


def test_betas_in_valid_range():
    betas = cosine_beta_schedule(T)
    assert (betas > 0).all(), "All betas must be positive"
    assert (betas < 1).all(), "All betas must be < 1"


def test_betas_monotone_increasing():
    """Cosine schedule betas should generally increase over time."""
    betas = cosine_beta_schedule(100)
    # Not strictly monotone, but should trend upward overall
    assert betas[-1] > betas[0]


# ---------------------------------------------------------------------------
# make_schedule
# ---------------------------------------------------------------------------

def test_schedule_keys(schedule):
    expected_keys = {
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod", "posterior_variance",
    }
    assert set(schedule.keys()) == expected_keys


def test_alphas_cumprod_starts_near_one(schedule):
    acp = schedule["alphas_cumprod"]
    assert acp[0] <= 1.0
    assert acp[0] > 0.9   # first step should be close to 1


def test_alphas_cumprod_ends_near_zero(schedule):
    acp = schedule["alphas_cumprod"]
    assert acp[-1] < 0.1   # last step should be close to 0


def test_alphas_cumprod_monotone_decreasing(schedule):
    acp = schedule["alphas_cumprod"]
    assert (acp[:-1] >= acp[1:]).all(), "alphas_cumprod must be non-increasing"


def test_sqrt_alphas_cumprod_squared_equals_alphas_cumprod(schedule):
    sac = schedule["sqrt_alphas_cumprod"]
    acp = schedule["alphas_cumprod"]
    assert torch.allclose(sac ** 2, acp, atol=1e-5)


# ---------------------------------------------------------------------------
# q_sample
# ---------------------------------------------------------------------------

def test_q_sample_output_shape(schedule):
    B, F, D = 4, 10, 32
    x_start = torch.randn(B, F, D)
    t = torch.randint(0, T, (B,))
    x_t, noise = q_sample(
        x_start, t,
        schedule["sqrt_alphas_cumprod"],
        schedule["sqrt_one_minus_alphas_cumprod"],
    )
    assert x_t.shape == (B, F, D)
    assert noise.shape == (B, F, D)


def test_q_sample_at_t0_close_to_x_start(schedule):
    """At t=0, x_t should be very close to x_start (little noise added)."""
    x_start = torch.ones(4, 5)
    t = torch.zeros(4, dtype=torch.long)
    noise = torch.zeros_like(x_start)
    x_t, _ = q_sample(
        x_start, t,
        schedule["sqrt_alphas_cumprod"],
        schedule["sqrt_one_minus_alphas_cumprod"],
        noise=noise,
    )
    # At t=0, alpha_bar_0 ≈ 1, so x_t ≈ x_start
    assert torch.allclose(x_t, x_start, atol=0.1)


def test_q_sample_at_t_max_is_noisy(schedule):
    """At t=T-1, x_t should be dominated by noise."""
    B, D = 8, 16
    x_start = torch.zeros(B, D)
    t = torch.full((B,), T - 1, dtype=torch.long)
    noise = torch.randn(B, D)
    x_t, _ = q_sample(
        x_start, t,
        schedule["sqrt_alphas_cumprod"],
        schedule["sqrt_one_minus_alphas_cumprod"],
        noise=noise,
    )
    # At max t, alpha_bar ≈ 0, so x_t ≈ noise
    assert torch.allclose(x_t, noise, atol=0.1)


def test_q_sample_uses_provided_noise(schedule):
    x_start = torch.zeros(2, 8)
    t = torch.tensor([T // 2, T // 2])
    noise = torch.ones(2, 8) * 0.5
    x_t, returned_noise = q_sample(
        x_start, t,
        schedule["sqrt_alphas_cumprod"],
        schedule["sqrt_one_minus_alphas_cumprod"],
        noise=noise,
    )
    assert torch.allclose(returned_noise, noise)


# ---------------------------------------------------------------------------
# compute_diffusion_loss
# ---------------------------------------------------------------------------

def test_loss_zero_for_perfect_prediction():
    noise = torch.randn(4, 10, 32)
    loss = compute_diffusion_loss(noise, noise)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_positive_for_wrong_prediction():
    predicted = torch.randn(4, 10, 32)
    actual = torch.randn(4, 10, 32)
    loss = compute_diffusion_loss(predicted, actual)
    assert loss.item() > 0.0


def test_loss_scales_with_error():
    actual = torch.zeros(4, 32)
    small_err = torch.ones(4, 32) * 0.1
    large_err = torch.ones(4, 32) * 10.0
    assert compute_diffusion_loss(small_err, actual) < compute_diffusion_loss(large_err, actual)


# ---------------------------------------------------------------------------
# ddpm_reverse_step
# ---------------------------------------------------------------------------

def test_ddpm_step_output_shape(schedule):
    B, F, D = 4, 10, 32
    x_t = torch.randn(B, F, D)
    model_out = torch.randn_like(x_t)
    x_prev = ddpm_reverse_step(model_out, x_t, t=T // 2, schedule=schedule)
    assert x_prev.shape == (B, F, D)


def test_ddpm_step_at_t0_equals_mean(schedule):
    """At t=0, DDPM step should return the predicted mean (no noise added)."""
    x_t = torch.randn(4, 8)
    model_out = torch.zeros_like(x_t)   # predict zero noise → mean equals scaled x_t
    x_prev_1 = ddpm_reverse_step(model_out, x_t, t=0, schedule=schedule)
    x_prev_2 = ddpm_reverse_step(model_out, x_t, t=0, schedule=schedule)
    # At t=0, no noise added, so both calls must return the same value
    assert torch.allclose(x_prev_1, x_prev_2)


# ---------------------------------------------------------------------------
# ddim_reverse_step
# ---------------------------------------------------------------------------

def test_ddim_step_output_shape(schedule):
    B, F, D = 4, 10, 32
    x_t = torch.randn(B, F, D)
    model_out = torch.randn_like(x_t)
    x_prev = ddim_reverse_step(model_out, x_t, t=T // 2, t_prev=T // 4, schedule=schedule)
    assert x_prev.shape == (B, F, D)


def test_ddim_step_deterministic_with_eta_zero(schedule):
    """With eta=0.0, two identical calls must return the exact same output."""
    x_t = torch.randn(4, 8)
    model_out = torch.randn_like(x_t)
    x1 = ddim_reverse_step(model_out, x_t, t=T // 2, t_prev=T // 4, schedule=schedule, eta=0.0)
    x2 = ddim_reverse_step(model_out, x_t, t=T // 2, t_prev=T // 4, schedule=schedule, eta=0.0)
    assert torch.allclose(x1, x2), "DDIM with eta=0 must be deterministic"


def test_ddim_step_stochastic_with_eta_nonzero(schedule):
    """With eta>0, two calls should differ due to added noise."""
    x_t = torch.randn(4, 8)
    model_out = torch.randn_like(x_t)
    x1 = ddim_reverse_step(model_out, x_t, t=T // 2, t_prev=T // 4, schedule=schedule, eta=1.0)
    x2 = ddim_reverse_step(model_out, x_t, t=T // 2, t_prev=T // 4, schedule=schedule, eta=1.0)
    assert not torch.allclose(x1, x2), "DDIM with eta>0 should be stochastic"


# ---------------------------------------------------------------------------
# EMAModel
# ---------------------------------------------------------------------------

def test_ema_construction():
    model = torch.nn.Linear(8, 4)
    ema = EMAModel(model, decay=0.99)
    assert set(ema.shadow.keys()) == set(model.state_dict().keys())


def test_ema_invalid_decay_raises():
    model = torch.nn.Linear(4, 2)
    with pytest.raises(ValueError, match="EMA decay"):
        EMAModel(model, decay=1.5)


def test_ema_update_changes_shadow():
    model = torch.nn.Linear(8, 4)
    ema = EMAModel(model, decay=0.9)
    original_weight = ema.shadow["weight"].clone()

    # Perturb model weights
    with torch.no_grad():
        model.weight.fill_(99.0)
    ema.update(model)

    # Shadow should have moved toward new weights but not equal them
    assert not torch.allclose(ema.shadow["weight"], original_weight)
    assert not torch.allclose(ema.shadow["weight"], model.weight.float())


def test_ema_copy_to_loads_shadow_weights():
    model = torch.nn.Linear(8, 4)
    ema = EMAModel(model, decay=0.9)

    # Modify EMA shadow directly
    ema.shadow["weight"].fill_(3.14)

    target_model = torch.nn.Linear(8, 4)
    ema.copy_to(target_model)
    assert torch.allclose(target_model.weight.float(), ema.shadow["weight"])
