"""
RLVR training with GRPO (Group Relative Policy Optimization).

Generates N candidate solutions per problem, executes each in an isolated
sandbox, normalizes rewards within the group (no value function needed),
and applies a policy gradient update with KL penalty.

This is the algorithm behind DeepSeek-R1, applied here at 350M scale on MBPP.

Key metrics to watch:
  - rlvr/mean_reward:  trending upward (noisy — that's normal for RL)
  - rlvr/pass_rate:    fraction of problems with at least one passing solution
  - rlvr/reward_std:   should decrease as solutions become more consistent
  - COLLAPSE SIGNAL:   reward_std → 0 AND mean_reward → 0 → degenerate policy
                       Fix: reduce kl_beta, add syntax partial reward

Usage:
    python src/training/rlvr_train.py --config configs/rlvr_code.yaml
"""

import json
import math
import os
import random
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import wandb
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.config import ModelConfig
from src.model.model import LLM
from sandbox.executor import execute_python_safe, syntax_check

ENC_MODULE = None  # Lazy-loaded to avoid import at module level


def _get_enc():
    global ENC_MODULE
    if ENC_MODULE is None:
        import tiktoken
        ENC_MODULE = tiktoken.get_encoding("gpt2")
    return ENC_MODULE


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class RLVRConfig:
    # Checkpoints
    checkpoint:     str = "checkpoints/cot/best.pt"
    ref_checkpoint: str = "checkpoints/v2/best.pt"   # Stable reference — pre-CoT/DPO
    output_dir:     str = "checkpoints/rlvr"

    # Data
    data_dir: str = "data/rlvr"

    # GRPO hyperparameters
    n_samples:   int   = 4      # Solutions generated per problem per step
    temperature: float = 0.8    # Sampling temperature
    kl_beta:     float = 0.04   # KL penalty — prevent collapse. Reduce if reward collapses.

    # Optimization — very conservative for RL
    learning_rate: float = 2e-6
    min_lr:        float = 5e-7
    max_epochs:    int   = 5
    beta1:         float = 0.9
    beta2:         float = 0.95
    weight_decay:  float = 0.01
    grad_clip:     float = 1.0
    grad_accum:    int   = 8

    # Sandbox
    sandbox_timeout_sec: int = 5
    sandbox_memory_mb:   int = 256

    # Partial reward for syntactically valid code (prevents collapse)
    syntax_reward: float = 0.1  # Added to reward if code parses; full reward=1.0 if tests pass

    # Logging & checkpointing
    log_interval:        int = 20
    eval_interval:       int = 100   # Run held-out MBPP test eval
    checkpoint_interval: int = 50

    wandb_project: str = "llm-v3-rlvr"
    max_new_tokens: int = 256
    seed: int = 42


# ── Generation ────────────────────────────────────────────────────────────────

_IM_END_STR = "<|im_end|>"

@torch.no_grad()
def _generate_one(
    model: LLM,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[int]:
    """Sample a single completion from the model. Returns generated token ids only."""
    enc = _get_enc()
    block_size = model.config.block_size

    ids = prompt_ids[-block_size:]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new_tokens):
        context = input_ids[:, -block_size:]
        logits, _ = model(context)
        next_logits = logits[0, -1] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == enc.eot_token:
            break

        generated.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )

        if _IM_END_STR in enc.decode(generated[-20:]):
            break

    return generated


def _extract_code(completion_text: str) -> str:
    """
    Pull Python code from the model's completion.
    Handles ```python ... ``` blocks or raw code.
    """
    # Try fenced code block first
    match = re.search(r"```(?:python)?\n(.*?)(?:```|$)", completion_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to raw text
    return completion_text.strip()


# ── Reward function ───────────────────────────────────────────────────────────

def _compute_reward(
    code: str,
    tests: list[str],
    solution: str,
    timeout: int,
    memory_mb: int,
    syntax_partial: float,
) -> float:
    """
    Reward signal:
      1.0  — all tests pass
      0.1  — syntax valid but tests fail (partial reward to prevent collapse)
      0.0  — syntax error or timeout
    """
    if not syntax_check(code):
        return 0.0

    result = execute_python_safe(code, tests, timeout_seconds=timeout, memory_limit_mb=memory_mb)
    if result["pass_rate"] >= 1.0:
        return 1.0

    return syntax_partial  # Syntactically valid but incorrect


# ── Sequence log-probs ────────────────────────────────────────────────────────

def _sequence_logprobs(
    model: LLM,
    input_ids: torch.Tensor,  # (1, T)
    response_start: int,
) -> torch.Tensor:
    """Sum of log-probs over generated tokens (response portion only). Returns scalar."""
    logits, _ = model(input_ids)                          # (1, T, V)
    log_probs = F.log_softmax(logits[0, :-1, :], dim=-1) # (T-1, V)
    targets   = input_ids[0, 1:]                          # (T-1,)

    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (T-1,)
    response_lp = token_lp[response_start - 1:]  # Slice to response tokens
    return response_lp.sum()


# ── GRPO step ─────────────────────────────────────────────────────────────────

def _grpo_step(
    policy: LLM,
    ref_model: LLM,
    problem: dict,
    cfg: RLVRConfig,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict]:
    enc = _get_enc()
    prompt_ids = enc.encode(problem["prompt"])
    response_start = len(prompt_ids)

    # Step 1: Generate n_samples candidate solutions
    solutions = []
    for _ in range(cfg.n_samples):
        gen_ids  = _generate_one(policy, prompt_ids, cfg.max_new_tokens, cfg.temperature, device)
        full_ids = prompt_ids + gen_ids
        code     = _extract_code(enc.decode(gen_ids))
        solutions.append({"full_ids": full_ids, "code": code})

    # Step 2: Score via sandbox execution
    rewards = torch.tensor([
        _compute_reward(
            s["code"], problem["tests"], problem["solution"],
            cfg.sandbox_timeout_sec, cfg.sandbox_memory_mb, cfg.syntax_reward,
        )
        for s in solutions
    ], dtype=torch.float32)

    metrics = {
        "rlvr/mean_reward": rewards.mean().item(),
        "rlvr/pass_rate":   (rewards >= 1.0).float().mean().item(),
        "rlvr/reward_std":  rewards.std().item(),
        "rlvr/syntax_rate": (rewards > 0).float().mean().item(),
    }

    # Step 3: Normalize within group (GRPO core insight — no value function needed)
    std = rewards.std()
    if std < 1e-8:
        # All rewards identical — no gradient signal, skip update
        metrics["rlvr/skipped"] = 1.0
        return None, metrics

    advantages = (rewards - rewards.mean()) / (std + 1e-8)

    # Step 4: Policy gradient loss + KL penalty for each solution
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    n_valid = 0

    for i, sol in enumerate(solutions):
        if abs(advantages[i].item()) < 1e-8:
            continue

        full_ids_t = torch.tensor([sol["full_ids"]], dtype=torch.long, device=device)
        full_ids_t = full_ids_t[:, :policy.config.block_size]  # Truncate to block_size

        policy_logps = _sequence_logprobs(policy, full_ids_t, response_start)
        with torch.no_grad():
            ref_logps = _sequence_logprobs(ref_model, full_ids_t, response_start)

        kl   = (policy_logps - ref_logps)  # Scalar, approximation of per-token KL
        loss = -(advantages[i].to(device) * policy_logps) + cfg.kl_beta * kl
        total_loss = total_loss + loss
        n_valid += 1

    if n_valid == 0:
        return None, metrics

    return total_loss / n_valid, metrics


# ── Eval on held-out MBPP test set ────────────────────────────────────────────

def _eval_mbpp_test(policy: LLM, test_problems: list[dict], cfg: RLVRConfig, device: torch.device) -> float:
    """Run greedy decode on held-out MBPP test problems. Returns pass rate."""
    enc = _get_enc()
    passed = 0
    policy.eval()
    with torch.no_grad():
        for problem in test_problems:
            prompt_ids = enc.encode(problem["prompt"])
            gen_ids    = _generate_one(policy, prompt_ids, cfg.max_new_tokens, temperature=0.0, device=device)
            code       = _extract_code(enc.decode(gen_ids))
            result     = execute_python_safe(
                code, problem["tests"],
                timeout_seconds=cfg.sandbox_timeout_sec,
                memory_limit_mb=cfg.sandbox_memory_mb,
            )
            if result["pass_rate"] >= 1.0:
                passed += 1
    policy.train()
    return passed / len(test_problems) if test_problems else 0.0


# ── Training loop ─────────────────────────────────────────────────────────────

def train_rlvr(cfg: RLVRConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load policy (CoT checkpoint)
    print(f"Loading policy from {cfg.checkpoint} ...")
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    model_cfg: ModelConfig = ckpt["model_config"]
    model_cfg.dropout = 0.0

    policy = LLM(model_cfg).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.train()

    # Load frozen reference (V2 — pre-CoT/DPO for stable KL anchor)
    print(f"Loading reference from {cfg.ref_checkpoint} ...")
    ref_ckpt  = torch.load(cfg.ref_checkpoint, map_location="cpu", weights_only=False)
    ref_model = LLM(model_cfg).to(device)
    ref_model.load_state_dict(ref_ckpt["model"])
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
        fused=True,
    )

    # Load problems
    data_path = Path(cfg.data_dir)
    with open(data_path / "train.json") as f:
        train_problems = json.load(f)
    with open(data_path / "test.json") as f:
        test_problems = json.load(f)
    print(f"  Train problems: {len(train_problems)}  |  Test (held-out): {len(test_problems)}")

    if cfg.wandb_project:
        wandb.init(project=cfg.wandb_project, config=vars(cfg))

    # Spot resilience
    output_path = Path(cfg.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _interrupted = [False]

    def _sigterm_handler(signum, frame):
        print("\nSIGTERM received — saving emergency checkpoint ...")
        _interrupted[0] = True

    signal.signal(signal.SIGTERM, _sigterm_handler)

    def _save(tag: str, step: int, best_pass_rate: float) -> None:
        path = output_path / f"{tag}.pt"
        torch.save({
            "model":         policy.state_dict(),
            "model_config":  model_cfg,
            "iter":          step,
            "mbpp_pass_rate": best_pass_rate,
            "config":        vars(cfg),
        }, path)
        print(f"  Saved {path}")

    best_pass_rate = 0.0
    step = 0

    for epoch in range(cfg.max_epochs):
        random.shuffle(train_problems)
        print(f"\n── Epoch {epoch+1}/{cfg.max_epochs} ──")

        optimizer.zero_grad(set_to_none=True)
        accum_metrics: dict[str, list] = {}
        accum_steps = 0

        for problem in train_problems:
            if _interrupted[0]:
                _save("emergency", step, best_pass_rate)
                return

            loss, metrics = _grpo_step(policy, ref_model, problem, cfg, device)

            for k, v in metrics.items():
                accum_metrics.setdefault(k, []).append(v)

            if loss is not None:
                (loss / cfg.grad_accum).backward()
                accum_steps += 1

            if accum_steps >= cfg.grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0

                if step % cfg.log_interval == 0:
                    avg = {k: sum(v) / len(v) for k, v in accum_metrics.items()}
                    print(
                        f"  step {step:4d} | reward {avg.get('rlvr/mean_reward', 0):.3f} | "
                        f"pass {avg.get('rlvr/pass_rate', 0):.3f} | "
                        f"std {avg.get('rlvr/reward_std', 0):.3f} | "
                        f"grad {grad_norm:.3f}"
                    )
                    if cfg.wandb_project:
                        wandb.log(avg, step=step)
                    accum_metrics = {}

                if step % cfg.eval_interval == 0 and step > 0:
                    pass_rate = _eval_mbpp_test(policy, test_problems, cfg, device)
                    print(f"  [eval] MBPP held-out pass rate: {pass_rate:.3f} ({pass_rate*100:.1f}%)")
                    if cfg.wandb_project:
                        wandb.log({"eval/mbpp_pass_rate": pass_rate}, step=step)
                    if pass_rate > best_pass_rate:
                        best_pass_rate = pass_rate
                        _save("best", step, best_pass_rate)

                if step % cfg.checkpoint_interval == 0 and step > 0:
                    _save("latest", step, best_pass_rate)

                step += 1

        # Flush remaining gradient accumulation at epoch end
        if accum_steps > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Final eval and checkpoint
    pass_rate = _eval_mbpp_test(policy, test_problems, cfg, device)
    print(f"\nFinal MBPP held-out pass rate: {pass_rate:.3f} ({pass_rate*100:.1f}%)")
    _save("latest", step, best_pass_rate)

    if cfg.wandb_project:
        wandb.log({"eval/mbpp_pass_rate_final": pass_rate}, step=step)
        wandb.finish()

    print(f"\nRLVR training complete. Best MBPP pass rate: {best_pass_rate:.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    cfg = RLVRConfig(**cfg_dict)
    train_rlvr(cfg)
