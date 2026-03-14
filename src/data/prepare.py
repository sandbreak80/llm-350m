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

    Memory-efficient: writes each document directly to disk as uint16 with no
    large in-memory buffers. Python list of ints is expensive (28 bytes/int);
    we convert each doc individually to avoid OOM.
    """
    import array as arr_mod

    output_dir.mkdir(parents=True, exist_ok=True)

    n_val_docs = int(FINEWEB_10BT_DOCS * val_fraction)  # ~48k docs
    n_train_docs = FINEWEB_10BT_DOCS - n_val_docs

    train_file = output_dir / "train.bin"
    val_file = output_dir / "val.bin"

    print("Streaming HuggingFaceFW/fineweb-edu (10BT sample)...")
    print("This will take ~2-3 hours.")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    train_tokens_written = 0
    val_tokens_written = 0

    # Open with large OS buffer — let the OS handle write batching
    with open(train_file, "wb", buffering=8 * 1024 * 1024) as train_fp, \
         open(val_file, "wb", buffering=8 * 1024 * 1024) as val_fp:

        for doc_idx, example in enumerate(tqdm(ds, total=FINEWEB_10BT_DOCS, desc="Tokenizing")):
            tokens = tokenize_text(example["text"])
            # Use array.array('H') — 2 bytes/token, no Python object overhead
            buf = arr_mod.array('H', tokens)

            if doc_idx < n_train_docs:
                buf.tofile(train_fp)
                train_tokens_written += len(tokens)
            else:
                buf.tofile(val_fp)
                val_tokens_written += len(tokens)

    print(f"train: {train_tokens_written:,} tokens ({train_file.stat().st_size / 1e9:.2f} GB)")
    print(f"val:   {val_tokens_written:,} tokens ({val_file.stat().st_size / 1e9:.2f} GB)")
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
