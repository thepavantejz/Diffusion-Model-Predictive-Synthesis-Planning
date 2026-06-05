"""Fine-tune ChemistryWorldModel (ReactionT5 + property heads + DynamicsDiffusion).

Two training phases:
  Phase 1 (property heads): train on reaction SMILES → property labels
  Phase 2 (dynamics):       train on trajectory data for F-step state prediction

Usage:
    python train/train_world_model.py \
        --model_config configs/model.yaml \
        --train_config configs/train.yaml \
        --data_dir data/processed/ \
        --out_dir checkpoints/world_model/ \
        --device cuda \
        --phase both

    # Fine-tune dynamics only (e.g. after new catalyst data)
    python train/train_world_model.py ... --phase dynamics
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SynthesisDataset
from data.preprocess import load_trajectories
from dmpsp.diffusion import EMAModel
from dmpsp.utils import resume_or_init, save_checkpoint, setup_logging
from dmpsp.world_model import build_world_model
from train.train_proposal import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ChemistryWorldModel.")
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--train_config", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--phase", default="both",
                        choices=["properties", "dynamics", "both"],
                        help="Which phase to train. Default: both.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    with open(args.model_config, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    device_str = args.device or train_cfg.get("device", "cpu")
    device = torch.device(device_str)
    max_steps = args.max_steps or int(train_cfg.get("max_steps", 100000))
    batch_size = args.batch_size or int(train_cfg.get("batch_size_world_model", 32))
    ema_decay = float(train_cfg.get("ema_decay", 0.99))
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))
    eval_every = int(train_cfg.get("eval_every", 1000))
    save_every = int(train_cfg.get("save_every", 5000))
    log_every = int(train_cfg.get("log_every", 100))

    model = build_world_model(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, max_steps)
    ema = EMAModel(model, decay=ema_decay)

    start_step = 0
    if args.resume:
        start_step = resume_or_init(args.out_dir, model, optimizer, device=device_str)

    train_path = args.data_dir / "trajectories_train.pkl"
    val_path = args.data_dir / "trajectories_val.pkl"
    if not train_path.exists():
        logger.error("Training data not found: %s", train_path)
        sys.exit(1)

    train_trajs = load_trajectories(train_path)
    val_trajs = load_trajectories(val_path) if val_path.exists() else []

    F = model_cfg.get("action_proposal", {}).get("horizon", 10)
    H = model_cfg.get("action_proposal", {}).get("history_len", 1)
    enc_dim = model_cfg.get("encoder", {}).get("hidden_dim", 256)

    train_ds = SynthesisDataset(
        train_trajs, "dynamics", horizon=F, history_len=H, encoder_hidden_dim=enc_dim
    )
    val_ds = SynthesisDataset(
        val_trajs, "dynamics", horizon=F, history_len=H, encoder_hidden_dim=enc_dim
    ) if val_trajs else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    logger.info(
        "Training ChemistryWorldModel [phase=%s]: %d steps, batch=%d, device=%s",
        args.phase, max_steps, batch_size, device,
    )

    model.train()
    step = start_step
    train_iter = iter(train_loader)
    running_loss = 0.0
    loss = torch.tensor(0.0)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        state_enc = batch["state_enc"].to(device)
        history_enc = batch["history_enc"].to(device)
        actions = batch["actions"].to(device)
        future_states = batch["future_states"].to(device)

        optimizer.zero_grad()

        if args.phase in ("dynamics", "both"):
            loss = model.dynamics_loss(future_states, state_enc, history_enc, actions)
        else:
            # Properties phase: skip if no reaction SMILES available in batch
            logger.debug("Properties phase not yet implemented in dataset — skipping.")
            step += 1
            continue

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
                for vb in val_loader:
                    vl = model.dynamics_loss(
                        vb["future_states"].to(device),
                        vb["state_enc"].to(device),
                        vb["history_enc"].to(device),
                        vb["actions"].to(device),
                    )
                    val_losses.append(vl.item())
            logger.info("step=%d  val_loss=%.4f", step, sum(val_losses) / len(val_losses))
            model.train()

        if step % save_every == 0:
            save_checkpoint(args.out_dir, model, optimizer, step, loss.item(), model_cfg, ema)

    save_checkpoint(args.out_dir, model, optimizer, step, loss.item(), model_cfg, ema)
    logger.info("Training complete. Saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
