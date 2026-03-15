"""
Pretraining loop for 350M LLM.

Usage:
    python src/training/train.py --config configs/pretrain_350m.yaml

Supports:
    - Single GPU (default)
    - DDP multi-GPU (torchrun --nproc_per_node=N src/training/train.py ...)
    - Spot instance resume from checkpoint
    - W&B logging
"""

import math
import os
import signal
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

import wandb
import yaml

from src.model.config import ModelConfig
from src.model.model import LLM
from src.training.config import TrainConfig


# ── Distributed helpers ──────────────────────────────────────────────────────

def is_ddp() -> bool:
    return int(os.environ.get("RANK", -1)) != -1

def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank

def destroy_ddp():
    dist.destroy_process_group()


# ── Dataset ──────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    def __init__(self, bin_path: Path, block_size: int):
        import numpy as np
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self) -> int:
        # Number of complete sequences — NOT token count.
        # Token count (~9.9B) causes randperm to allocate 79GB.
        return len(self.data) // self.block_size

    def __getitem__(self, idx: int):
        start = idx * self.block_size
        chunk = torch.from_numpy(self.data[start : start + self.block_size + 1].astype(int))
        return chunk[:-1].long(), chunk[1:].long()


# ── Learning rate schedule ───────────────────────────────────────────────────

def get_lr(it: int, cfg: "TrainConfig") -> float:
    # Linear warmup
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    # After decay: min lr
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # Cosine decay
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler_iter, val_loss, cfg: "TrainConfig", iteration: int):
    raw_model = model.module if isinstance(model, DDP) else model
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter": iteration,
        "val_loss": val_loss,
        "model_config": raw_model.config,
        "train_config": cfg,
    }
    path = Path(cfg.checkpoint_dir) / f"ckpt_{iteration:07d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    # Also save as latest.pt for easy resuming
    latest = Path(cfg.checkpoint_dir) / "latest.pt"
    torch.save(checkpoint, latest)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path: Path, model, optimizer) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iter"], checkpoint.get("val_loss", float("inf"))


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_loss(model, val_loader, eval_iters: int, device: torch.device) -> float:
    model.eval()
    losses = []
    val_iter = iter(val_loader)
    for _ in range(min(eval_iters, len(val_loader))):
        try:
            x, y = next(val_iter)
        except StopIteration:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("inf")


# ── Main training loop ───────────────────────────────────────────────────────

# ── Spot interruption handler ─────────────────────────────────────────────────
# AWS sends SIGTERM ~2 min before reclaiming a spot instance.
# We set a flag here (safe from signal context) and check it each iteration.
_shutdown_requested = False

def _sigterm_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\nSIGTERM received — spot interruption imminent. Will checkpoint at end of iteration.", flush=True)

signal.signal(signal.SIGTERM, _sigterm_handler)


def train(cfg: "TrainConfig"):
    use_ddp = is_ddp()
    rank = setup_ddp() if use_ddp else 0
    is_master = rank == 0
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed + rank)
    torch.set_float32_matmul_precision("high")

    # Model
    model_cfg = ModelConfig()
    model = LLM(model_cfg).to(device)
    if is_master:
        print(f"Model parameters: {model.num_params():,} (excl. embeddings)")

    # Compile (PyTorch 2.0+)
    if cfg.compile:
        model = torch.compile(model)

    if use_ddp:
        model = DDP(model, device_ids=[rank])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        fused=True,  # Faster fused AdamW on CUDA
    )

    # Data
    train_dataset = PretrainDataset(Path(cfg.data_dir) / "pretrain/train.bin", model_cfg.block_size)
    val_dataset = PretrainDataset(Path(cfg.data_dir) / "pretrain/val.bin", model_cfg.block_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train_iter = iter(train_loader)

    # Resume from checkpoint if available
    start_iter = 0
    best_val_loss = float("inf")
    resume_path = Path(cfg.checkpoint_dir) / "latest.pt"
    if resume_path.exists():
        start_iter, best_val_loss = load_checkpoint(resume_path, model, optimizer)
        if is_master:
            print(f"Resumed from iteration {start_iter}, val_loss={best_val_loss:.4f}")

    # W&B
    if is_master and cfg.wandb_project:
        wandb.init(project=cfg.wandb_project, config={**vars(cfg), **vars(model_cfg)}, resume="allow")

    scaler = torch.amp.GradScaler(enabled=False)  # bfloat16 doesn't need scaler

    model.train()
    t0 = time.time()

    for iteration in range(start_iter, cfg.max_iters):
        # Learning rate update
        lr = get_lr(iteration, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if use_ddp:
                model.require_backward_grad_sync = (micro_step == cfg.gradient_accumulation_steps - 1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / cfg.gradient_accumulation_steps

            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Logging
        if is_master and iteration % cfg.log_interval == 0:
            dt = time.time() - t0
            tokens_per_sec = (cfg.batch_size * cfg.gradient_accumulation_steps * model_cfg.block_size * cfg.log_interval) / dt
            print(f"iter {iteration:6d} | loss {loss.item() * cfg.gradient_accumulation_steps:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f} | {tokens_per_sec/1e3:.1f}K tok/s")
            if cfg.wandb_project:
                wandb.log({"train/loss": loss.item() * cfg.gradient_accumulation_steps, "train/lr": lr, "train/grad_norm": grad_norm, "train/tokens_per_sec": tokens_per_sec}, step=iteration)
            t0 = time.time()

        # Evaluation + checkpoint
        saved_this_iter = False
        if is_master and iteration % cfg.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, cfg.eval_iters, device)
            print(f"  val_loss: {val_loss:.4f}")
            if cfg.wandb_project:
                wandb.log({"val/loss": val_loss}, step=iteration)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, iteration, val_loss, cfg, iteration)
                saved_this_iter = True

        # Periodic checkpoint (for spot instance resilience) — skip if eval already saved
        if is_master and iteration % cfg.checkpoint_interval == 0 and iteration > start_iter and not saved_this_iter:
            save_checkpoint(model, optimizer, iteration, best_val_loss, cfg, iteration)

        # Spot interruption: SIGTERM received — save and exit cleanly
        if _shutdown_requested:
            if is_master:
                print("Saving emergency checkpoint before shutdown...", flush=True)
                save_checkpoint(model, optimizer, iteration, best_val_loss, cfg, iteration)
                if cfg.wandb_project:
                    wandb.finish()
            break

    if use_ddp:
        destroy_ddp()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = TrainConfig(**cfg_dict)
    train(cfg)
