"""
Dataset preparation — downloads and tokenizes pretraining + finetuning data.

Uses streaming mode for FineWeb-Edu to avoid the datasets library's
multiprocessing Arrow conversion (which dies on g5.xlarge).

Usage:
    python src/data/prepare.py --dataset pretrain --output_dir data/pretrain
    python src/data/prepare.py --dataset finetune --output_dir data/finetune
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


TOKENIZER = tiktoken.get_encoding("gpt2")
EOT = TOKENIZER.eot_token  # 50256
# Total docs in FineWeb-Edu 10BT sample
FINEWEB_10BT_DOCS = 9_672_101


def tokenize_text(text: str) -> list[int]:
    return TOKENIZER.encode_ordinary(text) + [EOT]


def prepare_pretrain(output_dir: Path, val_fraction: float = 0.005):
    """Stream FineWeb-Edu 10BT and tokenize directly to binary train/val files.

    Streaming avoids the datasets Arrow conversion multiprocessing that dies
    on this instance. Slightly slower per-doc but no subprocess crashes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Streaming HuggingFaceFW/fineweb-edu (10BT sample)...")
    print("This will take ~2-3 hours. Progress updates every 100k docs.")

    # Pre-allocate output files as memory-mapped arrays.
    # 10BT tokens ~ 20GB at uint16. We write train first, val at the end.
    # We'll collect val docs separately (small) and write train inline.
    n_val_docs = int(FINEWEB_10BT_DOCS * val_fraction)  # ~48k docs
    n_train_docs = FINEWEB_10BT_DOCS - n_val_docs

    # Two-pass approach: first pass streams and writes everything to a single
    # temp file, then we split. But that needs 2x space. Instead:
    # Simpler: reserve last val_fraction of docs for validation.
    train_tokens: list[int] = []
    val_tokens: list[int] = []
    train_written = 0
    val_written = 0

    # Use chunked writes to avoid holding all tokens in RAM
    CHUNK = 50_000_000  # flush every 50M tokens (~100MB)

    train_file = output_dir / "train.bin"
    val_file = output_dir / "val.bin"

    # Open files for appending as we go
    train_fp = open(train_file, "wb")
    val_fp = open(val_file, "wb")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    doc_idx = 0
    for example in tqdm(ds, total=FINEWEB_10BT_DOCS, desc="Tokenizing"):
        tokens = tokenize_text(example["text"])

        if doc_idx < n_train_docs:
            train_tokens.extend(tokens)
            if len(train_tokens) >= CHUNK:
                arr = np.array(train_tokens, dtype=np.uint16)
                arr.tofile(train_fp)
                train_written += len(train_tokens)
                train_tokens = []
        else:
            val_tokens.extend(tokens)

        doc_idx += 1

    # Flush remaining
    if train_tokens:
        arr = np.array(train_tokens, dtype=np.uint16)
        arr.tofile(train_fp)
        train_written += len(train_tokens)
    if val_tokens:
        arr = np.array(val_tokens, dtype=np.uint16)
        arr.tofile(val_fp)
        val_written += len(val_tokens)

    train_fp.close()
    val_fp.close()

    print(f"train: {train_written:,} tokens ({train_file.stat().st_size / 1e9:.2f} GB)")
    print(f"val:   {val_written:,} tokens ({val_file.stat().st_size / 1e9:.2f} GB)")
    print("Pretrain data ready.")


def prepare_finetune(output_dir: Path, num_proc: int = 4):
    """Prepare Alpaca instruction data with loss masks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/hf_cache/datasets")

    print("Loading yahma/alpaca-cleaned...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=cache_dir)

    # Anti-forgetting: stream 2500 samples from FineWeb (no Arrow conversion)
    print("Sampling 2500 FineWeb docs for anti-forgetting...")
    fw_stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    fw_samples = []
    for i, ex in enumerate(fw_stream):
        if i >= 5000:
            break
        fw_samples.append(ex)
    import random; random.seed(42); random.shuffle(fw_samples)
    fw_samples = fw_samples[:2500]

    def format_alpaca(example: dict) -> dict:
        if example.get("input", "").strip():
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        return {"prompt": prompt, "response": example["output"]}

    def tokenize_instruction(example: dict) -> dict:
        prompt_tokens = TOKENIZER.encode_ordinary(example["prompt"])
        response_tokens = TOKENIZER.encode_ordinary(example["response"]) + [EOT]
        tokens = prompt_tokens + response_tokens
        mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
        return {"tokens": tokens, "loss_mask": mask}

    # Process instruction data (small dataset, single-process fine)
    ds_fmt = ds.map(format_alpaca)
    ds_tok = ds_fmt.map(tokenize_instruction, num_proc=num_proc, remove_columns=ds_fmt.column_names)

    # Tokenize anti-forgetting samples inline
    fw_tok = [{"tokens": tokenize_text(ex["text"]), "loss_mask": None} for ex in tqdm(fw_samples, desc="Tokenizing FineWeb")]
    for ex in fw_tok:
        ex["loss_mask"] = [1] * len(ex["tokens"])

    # Combine and shuffle
    all_examples = list(ds_tok) + fw_tok
    random.seed(42)
    random.shuffle(all_examples)

    # Split train/val
    n_val = max(1, int(len(all_examples) * 0.02))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    for split, examples in [("train", train_examples), ("val", val_examples)]:
        filename = output_dir / f"{split}.jsonl"
        with open(filename, "w") as f:
            for ex in tqdm(examples, desc=f"Writing {split}"):
                f.write(json.dumps({"tokens": ex["tokens"], "loss_mask": ex["loss_mask"]}) + "\n")
        print(f"Saved {filename} ({len(examples):,} examples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pretrain", "finetune", "all"], default="all")
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    if args.dataset in ("pretrain", "all"):
        prepare_pretrain(args.output_dir / "pretrain")

    if args.dataset in ("finetune", "all"):
        prepare_finetune(args.output_dir / "finetune", num_proc=args.num_proc)
