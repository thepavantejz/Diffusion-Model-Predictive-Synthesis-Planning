"""Train ActionProposalDiffusion on synthesis trajectory data.

Usage:
    python train/train_proposal.py \
        --model_config configs/model.yaml \
        --train_config configs/train.yaml \
        --data_dir data/processed/ \
        --out_dir checkpoints/action_proposal/ \
        --device cuda

    # Overfit test (CPU, ~10 min)
    python train/train_proposal.py \
        --model_config configs/model.yaml \
        --train_config configs/train.yaml \
        --data_dir data/fixture/ \
        --out_dir checkpoints/overfit_proposal/ \
        --device cpu \
        --max_steps 2000 \
        --batch_size 4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SynthesisDataset
from data.preprocess import load_trajectories
from dmpsp.action_proposal import build_action_proposal
from dmpsp.diffusion import EMAModel
from dmpsp.encoder import build_encoder
from dmpsp.utils import resume_or_init, save_checkpoint, setup_logging

logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    opt_name = cfg.get("optimizer", "adam").lower()
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name!r}. Choose: adam, adamw")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    schedule = cfg.get("lr_schedule", "cosine").lower()
    warmup = int(cfg.get("warmup_steps", 500))
    if schedule == "cosine":
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, 1e-6, 1.0, total_iters=warmup),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup),
            ],
            milestones=[warmup],
        )
    if schedule == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    raise ValueError(f"Unsupported lr_schedule: {schedule!r}. Choose: cosine, constant")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ActionProposalDiffusion.")
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--train_config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True,
                        help="Directory containing trajectories_train.pkl and trajectories_val.pkl.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default=None,
                        help="Override device (cuda, cpu). Default: from train_config.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps from train_config.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from train_config.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in --out_dir.")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Override save_every from train_config.")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    # Load configs
    with open(args.model_config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    # Apply CLI overrides
    device_str = args.device or train_cfg.get("device", "cpu")
    device = torch.device(device_str)
    max_steps = args.max_steps or int(train_cfg.get("max_steps", 100000))
    batch_size = args.batch_size or int(train_cfg.get("batch_size_proposal", 64))
    ema_decay = float(train_cfg.get("ema_decay", 0.99))
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
    eval_every = int(train_cfg.get("eval_every", 1000))
    save_every = args.save_every or int(train_cfg.get("save_every", 5000))
    log_every = int(train_cfg.get("log_every", 100))

    # Build model and optimizer
    encoder = build_encoder(model_cfg.get("encoder", {})).to(device)
    model = build_action_proposal(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, max_steps)
    ema = EMAModel(model, decay=ema_decay)

    start_step = 0
    if args.resume:
        start_step = resume_or_init(args.out_dir, model, optimizer, device=device_str)

    # Load data
    train_path = args.data_dir / "trajectories_train.pkl"
    val_path = args.data_dir / "trajectories_val.pkl"
    if not train_path.exists():
        logger.error("Training data not found: %s", train_path)
        sys.exit(1)

    train_trajs = load_trajectories(train_path)
    val_trajs = load_trajectories(val_path) if val_path.exists() else []

    h = model_cfg.get("action_proposal", {}).get("history_len", 1)
    F = model_cfg.get("action_proposal", {}).get("horizon", 10)
    enc_dim = model_cfg.get("encoder", {}).get("hidden_dim", 256)

    train_ds = SynthesisDataset(train_trajs, "proposal", horizon=F, history_len=h, encoder_hidden_dim=enc_dim)
    val_ds = SynthesisDataset(val_trajs, "proposal", horizon=F, history_len=h, encoder_hidden_dim=enc_dim) if val_trajs else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    logger.info(
        "Training ActionProposalDiffusion: %d steps, batch=%d, device=%s",
        max_steps, batch_size, device,
    )

    model.train()
    step = start_step
    train_iter = iter(train_loader)
    running_loss = 0.0

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        state_enc = batch["state_enc"].to(device)
        history_enc = batch["history_enc"].to(device)
        actions = batch["actions"].to(device)

        optimizer.zero_grad()
        loss = model.training_loss(actions, state_enc, history_enc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        running_loss += loss.item()
        step += 1

        if step % log_every == 0:
            avg_loss = running_loss / log_every
            lr = optimizer.param_groups[0]["lr"]
            logger.info("step=%d  loss=%.4f  lr=%.2e", step, avg_loss, lr)
            running_loss = 0.0

        if val_loader and step % eval_every == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vl = model.training_loss(
                        vbatch["actions"].to(device),
                        vbatch["state_enc"].to(device),
                        vbatch["history_enc"].to(device),
                    )
                    val_losses.append(vl.item())
            logger.info("step=%d  val_loss=%.4f", step, sum(val_losses) / len(val_losses))
            model.train()

        if step % save_every == 0:
            save_checkpoint(
                args.out_dir, model, optimizer, step,
                loss=loss.item(), cfg=model_cfg, ema_model=ema,
            )

    # Final save
    save_checkpoint(args.out_dir, model, optimizer, step, loss=loss.item(), cfg=model_cfg, ema_model=ema)
    logger.info("Training complete. Final checkpoint saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
