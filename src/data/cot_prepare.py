"""
Prepare Chain-of-Thought SFT dataset for Phase 2.

Builds a ~60K example CoT dataset from three sources:
  1. NuminaMath-CoT   — 20K math problems with step-by-step solutions (wrapped in <think>)
  2. OpenHermes-2.5   — 30K filtered CoT responses (multi-step reasoning)
  3. FineWeb-Edu      — 10K anti-forgetting samples (streamed, no loss mask on prompt)

Output format matches InstructionDataset in finetune.py:
  {"tokens": [...], "loss_mask": [...]}
  loss_mask=1 for assistant tokens (including <think> blocks), 0 for prompt tokens.

Usage:
    python src/data/cot_prepare.py --output_dir data/cot
    python src/data/cot_prepare.py --output_dir data/cot --math_samples 20000 --hermes_samples 30000 --fineweb_samples 10000
"""

import argparse
import json
import random
from pathlib import Path

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

ENC = tiktoken.get_encoding("gpt2")

_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"
_EOT      = ENC.eot_token  # 50256


# ── Tokenization helpers ──────────────────────────────────────────────────────

def _tokenize_with_mask(prompt: str, response: str, max_length: int = 2048) -> dict | None:
    """
    Tokenize prompt + response. loss_mask=1 for response tokens only.
    Returns None if sequence exceeds max_length.
    """
    prompt_ids   = ENC.encode(prompt)
    response_ids = ENC.encode(response)
    all_ids      = prompt_ids + response_ids

    if len(all_ids) > max_length:
        return None

    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return {"tokens": all_ids, "loss_mask": loss_mask}


def _fineweb_record(text: str, max_length: int = 2048) -> dict | None:
    """
    Anti-forgetting sample — plain text continuation, all tokens are loss targets.
    No prompt/response structure; mimics pretraining signal.
    """
    ids = ENC.encode(text)
    if not ids:
        return None
    ids = ids[:max_length]
    return {"tokens": ids, "loss_mask": [1] * len(ids)}


# ── Source 1: NuminaMath-CoT ──────────────────────────────────────────────────

def _format_math_prompt(problem: str) -> str:
    return (
        f"{_IM_START}user\n"
        f"Solve the following math problem step by step:\n\n{problem.strip()}"
        f"{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )


def _format_math_response(solution: str) -> str:
    """Wrap solution in <think> tags — teaches model to show reasoning before answering."""
    return f"<think>\n{solution.strip()}\n</think>{_IM_END}"


def build_math_samples(max_samples: int, max_length: int) -> list[dict]:
    print(f"Loading AI-MO/NuminaMath-CoT ({max_samples:,} samples) ...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)

    records, skipped = [], 0
    for sample in tqdm(ds, total=max_samples, desc="  NuminaMath"):
        if len(records) >= max_samples:
            break
        problem  = sample.get("problem", "") or ""
        solution = sample.get("solution", "") or ""
        if not problem.strip() or not solution.strip():
            skipped += 1
            continue

        record = _tokenize_with_mask(
            _format_math_prompt(problem),
            _format_math_response(solution),
            max_length,
        )
        if record is None:
            skipped += 1
            continue
        records.append(record)

    print(f"  Got {len(records):,} math samples  (skipped {skipped:,})")
    return records


# ── Source 2: OpenHermes-2.5 CoT filter ──────────────────────────────────────

_COT_MARKERS = [
    "step 1", "step 2", "step 3",
    "first,", "second,", "third,",
    "let me", "let's think", "let us",
    "therefore,", "thus,", "hence,",
    "because", "since", "as a result",
    "1.", "2.", "3.",
]

def _is_cot_response(response: str) -> bool:
    """Heuristic: response shows explicit multi-step reasoning."""
    if len(response) < 150:
        return False
    lower = response.lower()
    marker_hits = sum(1 for m in _COT_MARKERS if m in lower)
    return marker_hits >= 2


def _format_hermes_prompt(conversations: list[dict]) -> tuple[str, str] | None:
    """
    Extract user/assistant turn from OpenHermes conversation format.
    Returns (prompt, response) or None if format is unexpected.
    """
    user_turn = next((c for c in conversations if c.get("from") == "human"), None)
    asst_turn = next((c for c in conversations if c.get("from") == "gpt"), None)
    if not user_turn or not asst_turn:
        return None

    prompt = (
        f"{_IM_START}user\n{user_turn['value'].strip()}{_IM_END}\n"
        f"{_IM_START}assistant\n"
    )
    response = asst_turn["value"].strip() + _IM_END
    return prompt, response


def build_hermes_cot_samples(max_samples: int, max_length: int) -> list[dict]:
    print(f"Loading teknium/OpenHermes-2.5 (filtering for CoT, target {max_samples:,}) ...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    records, checked, skipped = [], 0, 0
    for sample in tqdm(ds, desc="  OpenHermes CoT filter"):
        if len(records) >= max_samples:
            break
        checked += 1

        conversations = sample.get("conversations", [])
        result = _format_hermes_prompt(conversations)
        if result is None:
            skipped += 1
            continue

        prompt, response = result
        if not _is_cot_response(response):
            skipped += 1
            continue

        record = _tokenize_with_mask(prompt, response, max_length)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    print(f"  Got {len(records):,} CoT samples from {checked:,} checked  (skipped {skipped:,})")
    return records


# ── Source 3: FineWeb-Edu anti-forgetting blend ───────────────────────────────

def build_fineweb_samples(max_samples: int, max_length: int) -> list[dict]:
    print(f"Streaming FineWeb-Edu anti-forgetting blend ({max_samples:,} samples) ...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    records, skipped = [], 0
    for sample in tqdm(ds, total=max_samples, desc="  FineWeb-Edu"):
        if len(records) >= max_samples:
            break
        text = sample.get("text", "") or ""
        if len(text) < 100:
            skipped += 1
            continue
        record = _fineweb_record(text, max_length)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    print(f"  Got {len(records):,} anti-forgetting samples  (skipped {skipped:,})")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare_cot_data(
    output_dir: str,
    math_samples:    int = 20000,
    hermes_samples:  int = 30000,
    fineweb_samples: int = 10000,
    val_size:        int = 1000,
    max_length:      int = 2048,
    seed:            int = 42,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    all_records = []
    all_records += build_math_samples(math_samples, max_length)
    all_records += build_hermes_cot_samples(hermes_samples, max_length)
    all_records += build_fineweb_samples(fineweb_samples, max_length)

    print(f"\nTotal records: {len(all_records):,}")

    random.shuffle(all_records)
    val_records   = all_records[:val_size]
    train_records = all_records[val_size:]

    for split, data in [("train", train_records), ("val", val_records)]:
        out_file = output_path / f"{split}.jsonl"
        with open(out_file, "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {len(data):,} records → {out_file}")

    # Length stats
    lens = [len(r["tokens"]) for r in all_records]
    response_token_counts = [sum(r["loss_mask"]) for r in all_records]
    print(f"\nToken length stats:")
    print(f"  min={min(lens)}  max={max(lens)}  avg={sum(lens)//len(lens)}")
    print(f"  avg response tokens (loss targets): {sum(response_token_counts)//len(response_token_counts)}")
    print(f"\nDone. Data written to {output_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",      type=str, default="data/cot")
    parser.add_argument("--math_samples",    type=int, default=20000)
    parser.add_argument("--hermes_samples",  type=int, default=30000)
    parser.add_argument("--fineweb_samples", type=int, default=10000)
    parser.add_argument("--val_size",        type=int, default=1000)
    parser.add_argument("--max_length",      type=int, default=2048)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    prepare_cot_data(
        output_dir=args.output_dir,
        math_samples=args.math_samples,
        hermes_samples=args.hermes_samples,
        fineweb_samples=args.fineweb_samples,
        val_size=args.val_size,
        max_length=args.max_length,
        seed=args.seed,
    )
