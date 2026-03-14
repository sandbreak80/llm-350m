"""
Instruction finetuning (SFT) loop.

Usage:
    python src/training/finetune.py --config configs/finetune_instruct.yaml
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wandb
import yaml

from src.model.config import ModelConfig
from src.model.model import LLM


@dataclass
class FinetuneConfig:
    pretrain_checkpoint: str = "checkpoints/latest.pt"
    data_dir: str = "data/finetune"
    output_dir: str = "checkpoints/finetune"

    # Optimization (lower LR for finetuning)
    learning_rate: float = 2e-5
    min_lr: float = 3e-6
    warmup_iters: int = 0
    max_iters: int = 1500
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    batch_size: int = 4
    gradient_accumulation_steps: int = 32

    # Dropout enabled for finetuning (regularization)
    dropout: float = 0.1

    log_interval: int = 50
    eval_interval: int = 200
    eval_iters: int = 50

    wandb_project: str = "llm-350m-finetune"
    compile: bool = True
    seed: int = 42


class InstructionDataset(Dataset):
    def __init__(self, jsonl_path: Path, block_size: int):
        self.block_size = block_size
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        tokens = ex["tokens"]
        mask = ex["loss_mask"]

        # Truncate or pad to block_size
        tokens = tokens[: self.block_size + 1]
        mask = mask[: self.block_size + 1]

        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        m = torch.tensor(mask[:-1], dtype=torch.long)  # mask aligns with x (input positions)

        # Pad if shorter than block_size
        pad_len = self.block_size - len(x)
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)
            y = F.pad(y, (0, pad_len), value=-100)
            m = F.pad(m, (0, pad_len), value=0)

        return x, y, m


def get_lr(it: int, cfg: FinetuneConfig) -> float:
    if cfg.warmup_iters > 0 and it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = it / cfg.max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def finetune(cfg: FinetuneConfig):
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model
    checkpoint = torch.load(cfg.pretrain_checkpoint, map_location="cpu")
    model_cfg: ModelConfig = checkpoint["model_config"]
    model_cfg.dropout = cfg.dropout  # Enable dropout for finetuning

    model = LLM(model_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded pretrained checkpoint from {cfg.pretrain_checkpoint}")
    print(f"Pretrained val_loss: {checkpoint.get('val_loss', 'N/A')}")

    if cfg.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    train_dataset = InstructionDataset(Path(cfg.data_dir) / "train.jsonl", model_cfg.block_size)
    val_dataset = InstructionDataset(Path(cfg.data_dir) / "val.jsonl", model_cfg.block_size)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    train_iter = iter(train_loader)

    if cfg.wandb_project:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    best_val_loss = float("inf")
    model.train()

    for iteration in range(cfg.max_iters):
        lr = get_lr(iteration, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                x, y, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, mask = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y, loss_mask=mask)
                loss = loss / cfg.gradient_accumulation_steps

            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if iteration % cfg.log_interval == 0:
            train_loss = loss.item() * cfg.gradient_accumulation_steps
            print(f"iter {iteration:5d} | loss {train_loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f}")
            if cfg.wandb_project:
                wandb.log({"train/loss": train_loss, "train/lr": lr}, step=iteration)

        if iteration % cfg.eval_interval == 0:
            model.eval()
            losses = []
            val_iter = iter(val_loader)
            with torch.no_grad():
                for _ in range(min(cfg.eval_iters, len(val_loader))):
                    try:
                        x, y, mask = next(val_iter)
                    except StopIteration:
                        break
                    x, y, mask = x.to(device), y.to(device), mask.to(device)
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        _, loss = model(x, targets=y, loss_mask=mask)
                    losses.append(loss.item())
            val_loss = sum(losses) / len(losses) if losses else float("inf")
            print(f"  val_loss: {val_loss:.4f}")
            if cfg.wandb_project:
                wandb.log({"val/loss": val_loss}, step=iteration)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "model_config": model_cfg, "val_loss": val_loss, "iter": iteration}, f"{cfg.output_dir}/best.pt")
                print(f"  Saved best finetuned checkpoint (val_loss={val_loss:.4f})")
            model.train()

    print(f"Finetuning complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = FinetuneConfig(**cfg_dict)
    finetune(cfg)
