"""
Prepare RLVR dataset from MBPP (Mostly Basic Python Problems).

MBPP is simpler than HumanEval and better suited for 350M scale — problems
are short, test cases are explicit, and the solution space is tractable.

Each record contains:
  - prompt:     ChatML-formatted instruction (problem description only — no solution shown)
  - tests:      List of assert strings used as the reward signal
  - solution:   Reference solution (eval only, NEVER shown during training)
  - task_id:    For deduplication and tracking

Usage:
    python src/data/rlvr_prepare.py --output_dir data/rlvr
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"


def _format_code_prompt(description: str) -> str:
    return (
        f"{_IM_START}user\n"
        f"Write a Python function to solve the following problem. "
        f"Return only the function with no explanation.\n\n"
        f"{description.strip()}"
        f"{_IM_END}\n"
        f"{_IM_START}assistant\n"
        "```python\n"
    )


def prepare_rlvr_data(output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading google-research-datasets/mbpp (sanitized split) ...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
    print(f"  {len(ds):,} problems available")

    records = []
    skipped = 0

    for item in ds:
        description = item.get("text", "") or ""
        code        = item.get("code", "") or ""
        test_list   = item.get("test_list", []) or []
        task_id     = item.get("task_id", 0)

        if not description.strip() or not test_list:
            skipped += 1
            continue

        records.append({
            "task_id":  task_id,
            "prompt":   _format_code_prompt(description),
            "tests":    test_list,
            "solution": code,  # Never shown during training
        })

    print(f"  {len(records):,} usable problems  (skipped {skipped:,})")

    # Write all problems — RLVR doesn't need a traditional train/val split
    # (the reward signal IS execution; held-out eval uses HumanEval via code_eval.py)
    # We do hold out the last 50 task_ids as a held-out MBPP test set
    records_sorted = sorted(records, key=lambda r: r["task_id"])
    train_records  = records_sorted[:-50]
    test_records   = records_sorted[-50:]

    for split, data in [("train", train_records), ("test", test_records)]:
        out_file = output_path / f"{split}.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Wrote {len(data):,} records → {out_file}")

    print(f"\nDone. Data written to {output_path}/")
    print(f"  Train problems used for RLVR training: {len(train_records)}")
    print(f"  Held-out test problems (MBPP pass rate): {len(test_records)}")
    print(f"  HumanEval (164 problems) used for final generalization eval via code_eval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/rlvr")
    args = parser.parse_args()
    prepare_rlvr_data(args.output_dir)
