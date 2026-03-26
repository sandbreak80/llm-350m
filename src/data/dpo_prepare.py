"""
Prepare DPO preference pair dataset from intel/orca_dpo_pairs.

Downloads ~12,859 GPT-4 quality preference pairs, formats them in ChatML,
tokenizes with GPT-2 tiktoken, and writes train/val JSONL files ready for
dpo_train.py.

Each JSONL record contains:
  - chosen_ids:   token ids for full sequence (prompt + chosen response)
  - rejected_ids: token ids for full sequence (prompt + rejected response)
  - response_start: index into token list where response begins (for loss mask)

Usage:
    python src/data/dpo_prepare.py --output_dir data/dpo
    python src/data/dpo_prepare.py --output_dir data/dpo --max_samples 10000 --val_size 500
"""

import argparse
import json
import random
from pathlib import Path

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

ENC = tiktoken.get_encoding("gpt2")

# ChatML tokens (tokenized as normal text by GPT-2 encoder)
_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"


def _format_prompt(system: str, question: str) -> str:
    """Format the prompt portion (everything before the assistant response)."""
    parts = []
    if system and system.strip():
        parts.append(f"{_IM_START}system\n{system.strip()}{_IM_END}\n")
    parts.append(f"{_IM_START}user\n{question.strip()}{_IM_END}\n")
    parts.append(f"{_IM_START}assistant\n")
    return "".join(parts)


def _format_response(response: str) -> str:
    return response.strip() + _IM_END


def _tokenize_pair(
    system: str,
    question: str,
    chosen: str,
    rejected: str,
    max_length: int = 1024,
) -> dict | None:
    """
    Tokenize a preference pair.

    Returns a dict with chosen_ids, rejected_ids, response_start, or None
    if either sequence exceeds max_length (skip rather than truncate chosen/rejected).
    """
    prompt_text    = _format_prompt(system, question)
    chosen_text    = _format_response(chosen)
    rejected_text  = _format_response(rejected)

    prompt_ids    = ENC.encode(prompt_text)
    chosen_ids    = ENC.encode(chosen_text)
    rejected_ids  = ENC.encode(rejected_text)

    full_chosen   = prompt_ids + chosen_ids
    full_rejected = prompt_ids + rejected_ids

    # Skip pairs where either full sequence is too long — don't truncate responses
    if len(full_chosen) > max_length or len(full_rejected) > max_length:
        return None

    return {
        "chosen_ids":    full_chosen,
        "rejected_ids":  full_rejected,
        "response_start": len(prompt_ids),  # Index where response tokens begin
    }


def prepare_dpo_data(
    output_dir: str,
    max_samples: int = 10000,
    val_size: int = 500,
    max_length: int = 1024,
    seed: int = 42,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading intel/orca_dpo_pairs ...")
    ds = load_dataset("Intel/orca_dpo_pairs", split="train")
    print(f"  {len(ds):,} total samples available")

    random.seed(seed)
    indices = list(range(len(ds)))
    random.shuffle(indices)
    indices = indices[:max_samples]

    records = []
    skipped = 0

    for idx in tqdm(indices, desc="Tokenizing"):
        sample = ds[idx]
        system   = sample.get("system", "") or ""
        question = sample.get("question", "") or ""
        chosen   = sample.get("chosen", "") or ""
        rejected = sample.get("rejected", "") or ""

        if not question.strip() or not chosen.strip() or not rejected.strip():
            skipped += 1
            continue

        record = _tokenize_pair(system, question, chosen, rejected, max_length)
        if record is None:
            skipped += 1
            continue

        records.append(record)

    print(f"\n  Processed: {len(records):,} pairs  |  Skipped (too long / empty): {skipped:,}")

    # Split
    random.shuffle(records)
    val_records   = records[:val_size]
    train_records = records[val_size:]

    for split, data in [("train", train_records), ("val", val_records)]:
        out_file = output_path / f"{split}.jsonl"
        with open(out_file, "w") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")
        print(f"  Wrote {len(data):,} records → {out_file}")

    # Stats
    chosen_lens   = [len(r["chosen_ids"])   for r in records]
    rejected_lens = [len(r["rejected_ids"]) for r in records]
    print(f"\nLength stats (tokens):")
    print(f"  chosen:   min={min(chosen_lens)}  max={max(chosen_lens)}  avg={sum(chosen_lens)//len(chosen_lens)}")
    print(f"  rejected: min={min(rejected_lens)}  max={max(rejected_lens)}  avg={sum(rejected_lens)//len(rejected_lens)}")
    print(f"\nDone. Data written to {output_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  type=str, default="data/dpo")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--val_size",    type=int, default=500)
    parser.add_argument("--max_length",  type=int, default=1024)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    prepare_dpo_data(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        val_size=args.val_size,
        max_length=args.max_length,
        seed=args.seed,
    )
