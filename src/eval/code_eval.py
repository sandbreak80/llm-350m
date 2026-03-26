"""
HumanEval pass@k evaluation for instruction-tuned checkpoints.

Loads problems from the openai_humaneval HuggingFace dataset, generates
N completions per problem using the model, executes each in an isolated
subprocess, and computes pass@1 (and pass@k for k <= n_samples).

Usage:
    from src.eval.code_eval import eval_humaneval
    results = eval_humaneval(model, n_samples=20, temperature=0.8)
    # {"humaneval/pass@1": 0.073, "humaneval/pass@10": 0.21, ...}

Notes:
    - Prompts are formatted in ChatML (matches V2/V3 training format).
    - Generation stops at <|im_end|> or <|endoftext|> or max_new_tokens.
    - Code execution uses subprocess with a hard timeout — never inline exec().
    - Run on CPU (same as all other evals). Slow but safe.
"""

import os
import re
import sys
import math
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.model import LLM

ENC = tiktoken.get_encoding("gpt2")

# GPT-2 EOS token id
_EOT_ID = ENC.eot_token  # 50256

# ChatML stop string — stop generation when this appears in decoded output
_IM_END = "<|im_end|>"

# Sandbox limits
_EXEC_TIMEOUT_SEC = 10
_MAX_NEW_TOKENS = 384


# ── Prompt formatting ─────────────────────────────────────────────────────────

def _format_humaneval_prompt(problem_prompt: str) -> str:
    """
    Wrap a HumanEval function stub in a ChatML user/assistant turn.
    The model should complete the function body starting after the stub.
    """
    instruction = (
        "Complete the following Python function. "
        "Return only the completed function with no explanation.\n\n"
        f"```python\n{problem_prompt.rstrip()}\n```"
    )
    return (
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n```python\n{problem_prompt.rstrip()}\n"
    )


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _generate(
    model: LLM,
    prompt: str,
    max_new_tokens: int = _MAX_NEW_TOKENS,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """
    Sample a single completion from the model given a text prompt.
    Stops at <|endoftext|>, <|im_end|>, or max_new_tokens.
    """
    tokens = ENC.encode(prompt)
    block_size = model.config.block_size

    # Truncate prompt from left if needed, keep as many tokens as possible
    if len(tokens) > block_size - 1:
        tokens = tokens[-(block_size - 1):]

    input_ids = torch.tensor([tokens], dtype=torch.long)
    generated = []

    for _ in range(max_new_tokens):
        # Only pass the last block_size tokens
        context = input_ids[:, -block_size:]
        logits, _ = model(context)
        next_logits = logits[0, -1, :]  # (vocab_size,)

        if temperature == 0.0:
            next_token = next_logits.argmax().item()
        else:
            next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)

            # Top-p nucleus sampling
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            # Remove tokens past the nucleus
            sorted_probs[cumsum - sorted_probs > top_p] = 0.0
            sorted_probs /= sorted_probs.sum()
            sampled = torch.multinomial(sorted_probs, num_samples=1).item()
            next_token = sorted_idx[sampled].item()

        if next_token == _EOT_ID:
            break

        generated.append(next_token)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1
        )

        # Stop on <|im_end|> in decoded text (multi-token string)
        decoded_so_far = ENC.decode(generated)
        if _IM_END in decoded_so_far:
            break

    completion = ENC.decode(generated)
    # Strip the im_end suffix if present
    if _IM_END in completion:
        completion = completion[: completion.index(_IM_END)]
    return completion


# ── Code extraction ───────────────────────────────────────────────────────────

def _extract_function_body(completion: str, prompt: str) -> str:
    """
    Extract the function implementation from a model completion.
    The model is prompted with the stub already in place, so we grab
    everything after the last line of the prompt.

    Falls back to returning the full completion if extraction fails.
    """
    # Try to find a clean code block ending
    if "```" in completion:
        # Extract content between first ``` and closing ```
        parts = completion.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("python"):
                code = code[6:]
            return code.strip()

    # Return whatever was generated — execution will validate it
    return completion.strip()


# ── Sandbox execution ─────────────────────────────────────────────────────────

def _execute_solution(
    solution_code: str,
    test_code: str,
    timeout: int = _EXEC_TIMEOUT_SEC,
) -> bool:
    """
    Execute solution_code + test_code in an isolated subprocess.
    Returns True if the subprocess exits with code 0, False otherwise.

    IMPORTANT: Never use eval() or exec() inline — always use subprocess.
    """
    full_code = solution_code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    ) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── pass@k estimator ─────────────────────────────────────────────────────────

def _pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator (Chen et al. 2021).
    n = total samples, c = correct samples, k = k in pass@k.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


# ── Main eval function ────────────────────────────────────────────────────────

def eval_humaneval(
    model: LLM,
    n_samples: int = 20,
    temperature: float = 0.8,
    top_p: float = 0.95,
    k_values: Optional[list[int]] = None,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate pass@k on the full HumanEval benchmark (164 problems).

    Args:
        model:       Loaded LLM in eval mode on CPU.
        n_samples:   Number of completions to generate per problem.
        temperature: Sampling temperature (0.0 = greedy).
        top_p:       Nucleus sampling threshold.
        k_values:    List of k values for pass@k. Defaults to [1, 10].
        verbose:     Print progress per problem.

    Returns:
        Dict of {"humaneval/pass@1": float, "humaneval/pass@10": float, ...}
    """
    from datasets import load_dataset

    if k_values is None:
        k_values = [1, 10]

    # Clamp k values to n_samples
    k_values = [k for k in k_values if k <= n_samples]

    ds = load_dataset("openai_humaneval", split="test")
    problems = list(ds)

    total = len(problems)
    per_problem_results: list[dict] = []

    for idx, problem in enumerate(problems):
        task_id = problem["task_id"]
        prompt = problem["prompt"]          # Function stub
        test = problem["test"]              # assert-based test suite
        entry_point = problem["entry_point"]

        formatted_prompt = _format_humaneval_prompt(prompt)

        correct = 0
        for sample_idx in range(n_samples):
            completion = _generate(
                model, formatted_prompt,
                temperature=temperature, top_p=top_p,
            )
            solution = _extract_function_body(completion, prompt)

            # Build the full solution: stub + body + tests
            # The test code calls check(entry_point) — standard HumanEval format
            full_solution = prompt + "\n" + solution
            passed = _execute_solution(full_solution, test)
            if passed:
                correct += 1

        per_problem_results.append({"n": n_samples, "c": correct})

        if verbose:
            pass_rate = correct / n_samples
            print(
                f"  [{idx+1:3d}/{total}] {task_id:<30} "
                f"pass={correct}/{n_samples} ({pass_rate:.0%})",
                flush=True,
            )

    # Aggregate pass@k across all problems
    results = {}
    for k in k_values:
        scores = [_pass_at_k(r["n"], r["c"], k) for r in per_problem_results]
        pass_at_k_score = sum(scores) / len(scores)
        results[f"humaneval/pass@{k}"] = pass_at_k_score

    # Also store raw per-problem correct counts for debugging
    results["humaneval/n_problems"] = total
    results["humaneval/n_samples_per_problem"] = n_samples
    results["humaneval/total_correct"] = sum(r["c"] for r in per_problem_results)

    return results
