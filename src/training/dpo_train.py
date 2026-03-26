"""
DPO (Direct Preference Optimization) training loop.

Trains a policy model to prefer chosen responses over rejected ones using a
frozen reference model (V2 checkpoint). No reward model needed.

Loss: -log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x))
                - β * (log π_θ(y_l|x) - log π_ref(y_l|x)))

Key metric to watch: dpo/reward_margin — must trend positive and stable.
If flat: halve lr. If spike-then-collapse: reduce beta.

Usage:
    python src/training/dpo_train.py --config configs/dpo.yaml
"""

import json
import math
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import wandb
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.config import ModelConfig
from src.model.model import LLM


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class DPOConfig:
    # Checkpoints
    checkpoint:     str = "checkpoints/v2/best.pt"
    ref_checkpoint: str = "checkpoints/v2/best.pt"  # Same file, loaded frozen separately
    output_dir:     str = "checkpoints/dpo"

    # Data
    data_dir:   str = "data/dpo"
    max_length: int = 1024

    # DPO hyperparameters
    beta: float = 0.1  # KL penalty — how much to deviate from ref. Raise to 0.2 if margin is flat.

    # Optimization — DPO is very LR-sensitive, keep conservative
    learning_rate: float = 5e-6
    min_lr:        float = 5e-7
    max_iters:     int   = 800
    beta1:         float = 0.9
    beta2:         float = 0.95
    weight_decay:  float = 0.01
    grad_clip:     float = 1.0

    batch_size:                 int = 2
    gradient_accumulation_steps: int = 16  # Effective batch = 32 sequences

    # Logging & checkpointing
    log_interval:        int = 10
    eval_interval:       int = 50
    eval_iters:          int = 25
    checkpoint_interval: int = 100

    wandb_project: str = "llm-v3-dpo"
    seed:          int = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class DPODataset(Dataset):
    def __init__(self, jsonl_path: Path, max_length: int):
        self.max_length = max_length
        self.examples = []
        with open(jsonl_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        chosen_ids   = ex["chosen_ids"][: self.max_length]
        rejected_ids = ex["rejected_ids"][: self.max_length]
        response_start = ex["response_start"]

        def make_mask(ids: list[int]) -> list[int]:
            """1 for response tokens, 0 for prompt tokens."""
            return [0] * min(response_start, len(ids)) + [1] * max(0, len(ids) - response_start)

        return {
            "chosen_ids":    chosen_ids,
            "rejected_ids":  rejected_ids,
            "chosen_mask":   make_mask(chosen_ids),
            "rejected_mask": make_mask(rejected_ids),
        }


def _collate(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences to the longest in the batch."""
    def pad(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        padded = [s + [pad_val] * (max_len - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long)

    return {
        "chosen_ids":    pad([b["chosen_ids"]    for b in batch], pad_id),
        "rejected_ids":  pad([b["rejected_ids"]  for b in batch], pad_id),
        "chosen_mask":   pad([b["chosen_mask"]   for b in batch], 0),
        "rejected_mask": pad([b["rejected_mask"] for b in batch], 0),
    }


# ── DPO loss ──────────────────────────────────────────────────────────────────

def _sequence_logprobs(
    model: LLM,
    input_ids: torch.Tensor,   # (B, T)
    response_mask: torch.Tensor,  # (B, T)  1 = response token
) -> torch.Tensor:
    """
    Compute sum of log-probs over response tokens only.
    Returns shape (B,).
    """
    logits, _ = model(input_ids)                               # (B, T, V)
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)      # (B, T-1, V)
    targets   = input_ids[:, 1:]                               # (B, T-1)
    mask      = response_mask[:, 1:].float()                   # (B, T-1) shifted

    token_log_probs = log_probs.gather(
        2, targets.unsqueeze(-1)
    ).squeeze(-1)                                              # (B, T-1)

    return (token_log_probs * mask).sum(dim=-1)                # (B,)


def dpo_loss(
    policy_chosen_logps:   torch.Tensor,   # (B,)
    policy_rejected_logps: torch.Tensor,   # (B,)
    ref_chosen_logps:      torch.Tensor,   # (B,)
    ref_rejected_logps:    torch.Tensor,   # (B,)
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    chosen_rewards   = beta * (policy_chosen_logps   - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    metrics = {
        "dpo/loss":            loss.item(),
        "dpo/chosen_reward":   chosen_rewards.mean().item(),
        "dpo/rejected_reward": rejected_rewards.mean().item(),
        "dpo/reward_margin":   (chosen_rewards - rejected_rewards).mean().item(),
    }
    return loss, metrics


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(it: int, cfg: DPOConfig) -> float:
    if it > cfg.max_iters:
        return cfg.min_lr
    decay_ratio = it / cfg.max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_dpo(cfg: DPOConfig) -> None:
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load policy model (will be trained)
    print(f"Loading policy model from {cfg.checkpoint} ...")
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    model_cfg: ModelConfig = ckpt["model_config"]
    model_cfg.dropout = 0.0  # No dropout for DPO

    policy = LLM(model_cfg).to(device)
    policy.load_state_dict(ckpt["model"])
    print(f"  Policy val_loss at start: {ckpt.get('val_loss', 'N/A')}")

    # ── Load reference model (frozen, never updated)
    print(f"Loading reference model from {cfg.ref_checkpoint} ...")
    ref_ckpt = torch.load(cfg.ref_checkpoint, map_location="cpu", weights_only=False)
    ref_model = LLM(model_cfg).to(device)
    ref_model.load_state_dict(ref_ckpt["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print(f"  Reference model frozen.")

    # ── Optimizer (policy only)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    # ── Data
    data_path = Path(cfg.data_dir)
    train_ds = DPODataset(data_path / "train.jsonl", cfg.max_length)
    val_ds   = DPODataset(data_path / "val.jsonl",   cfg.max_length)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )
    train_iter = iter(train_loader)
    print(f"  Train: {len(train_ds):,} pairs  |  Val: {len(val_ds):,} pairs")

    # ── W&B
    if cfg.wandb_project:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # ── Spot instance resilience
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _interrupted = [False]

    def _sigterm_handler(signum, frame):
        print("\nSIGTERM received — saving emergency checkpoint ...")
        _interrupted[0] = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    def _save(tag: str, iteration: int, reward_margin: float) -> None:
        path = output_path / f"{tag}.pt"
        torch.save({
            "model":        policy.state_dict(),
            "model_config": model_cfg,
            "iter":         iteration,
            "reward_margin": reward_margin,
            "config":       vars(cfg),
        }, path)
        print(f"  Saved {path}")

    # ── Training
    best_margin = -float("inf")
    policy.train()

    for iteration in range(cfg.max_iters):
        if _interrupted[0]:
            _save("emergency", iteration, best_margin)
            break

        lr = get_lr(iteration, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_metrics: dict[str, float] = {}

        for micro_step in range(cfg.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            chosen_ids    = batch["chosen_ids"].to(device,    non_blocking=True)
            rejected_ids  = batch["rejected_ids"].to(device,  non_blocking=True)
            chosen_mask   = batch["chosen_mask"].to(device,   non_blocking=True)
            rejected_mask = batch["rejected_mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Reference logprobs — no grad, frozen
                with torch.no_grad():
                    ref_chosen_logps   = _sequence_logprobs(ref_model, chosen_ids,   chosen_mask)
                    ref_rejected_logps = _sequence_logprobs(ref_model, rejected_ids, rejected_mask)

                # Policy logprobs — with grad
                policy_chosen_logps   = _sequence_logprobs(policy, chosen_ids,   chosen_mask)
                policy_rejected_logps = _sequence_logprobs(policy, rejected_ids, rejected_mask)

                loss, metrics = dpo_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    ref_chosen_logps,    ref_rejected_logps,
                    beta=cfg.beta,
                )
                loss = loss / cfg.gradient_accumulation_steps

            loss.backward()

            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + v / cfg.gradient_accumulation_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()

        if iteration % cfg.log_interval == 0:
            print(
                f"iter {iteration:4d} | loss {accum_metrics.get('dpo/loss', 0):.4f} | "
                f"margin {accum_metrics.get('dpo/reward_margin', 0):.4f} | "
                f"lr {lr:.2e} | grad {grad_norm:.3f}"
            )
            if cfg.wandb_project:
                wandb.log({**accum_metrics, "train/lr": lr, "train/grad_norm": grad_norm}, step=iteration)

        # ── Eval
        if iteration % cfg.eval_interval == 0:
            policy.eval()
            val_metrics: dict[str, list] = {k: [] for k in ["dpo/loss", "dpo/reward_margin",
                                                              "dpo/chosen_reward", "dpo/rejected_reward"]}
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    if i >= cfg.eval_iters:
                        break
                    chosen_ids    = batch["chosen_ids"].to(device)
                    rejected_ids  = batch["rejected_ids"].to(device)
                    chosen_mask   = batch["chosen_mask"].to(device)
                    rejected_mask = batch["rejected_mask"].to(device)

                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        ref_chosen_logps   = _sequence_logprobs(ref_model, chosen_ids,   chosen_mask)
                        ref_rejected_logps = _sequence_logprobs(ref_model, rejected_ids, rejected_mask)
                        pol_chosen_logps   = _sequence_logprobs(policy,    chosen_ids,   chosen_mask)
                        pol_rejected_logps = _sequence_logprobs(policy,    rejected_ids, rejected_mask)
                        _, m = dpo_loss(pol_chosen_logps, pol_rejected_logps,
                                        ref_chosen_logps,  ref_rejected_logps, beta=cfg.beta)

                    for k in val_metrics:
                        val_metrics[k].append(m[k])

            avg = {k: sum(v) / len(v) for k, v in val_metrics.items() if v}
            reward_margin = avg.get("dpo/reward_margin", -float("inf"))
            print(
                f"  [val] loss={avg.get('dpo/loss', 0):.4f} | "
                f"margin={reward_margin:.4f} | "
                f"chosen={avg.get('dpo/chosen_reward', 0):.4f} | "
                f"rejected={avg.get('dpo/rejected_reward', 0):.4f}"
            )
            if cfg.wandb_project:
                wandb.log({f"val/{k.split('/')[1]}": v for k, v in avg.items()}, step=iteration)

            if reward_margin > best_margin:
                best_margin = reward_margin
                _save("best", iteration, reward_margin)

            policy.train()

        if iteration % cfg.checkpoint_interval == 0 and iteration > 0:
            _save("latest", iteration, best_margin)

    # Final checkpoint
    _save("latest", cfg.max_iters, best_margin)
    print(f"\nDPO training complete. Best reward_margin: {best_margin:.4f}")
    if cfg.wandb_project:
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = DPOConfig(**cfg_dict)
    train_dpo(cfg)
