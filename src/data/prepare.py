"""
Dataset preparation — downloads and tokenizes pretraining + finetuning data.

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


def tokenize_text(text: str) -> list[int]:
    return TOKENIZER.encode_ordinary(text) + [EOT]


def prepare_pretrain(output_dir: Path, num_proc: int = 8):
    """Download FineWeb-Edu 10BT sample and tokenize to binary shards."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/hf_cache/datasets")

    print("Loading HuggingFaceFW/fineweb-edu (10BT sample)...")
    # num_proc=2 for download/Arrow conversion — more causes OOM on 16GB RAM
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        num_proc=2,
        cache_dir=cache_dir,
    )

    # Tokenize in parallel (CPU-bound, safe to use more workers here)
    def tokenize_batch(examples):
        tokens = [tokenize_text(t) for t in examples["text"]]
        lengths = [len(t) for t in tokens]
        return {"tokens": tokens, "length": lengths}

    ds = ds.map(tokenize_batch, batched=True, num_proc=num_proc, remove_columns=ds.column_names)

    # Split train/val
    ds = ds.train_test_split(test_size=0.005, seed=42)

    for split, dataset in ds.items():
        total_tokens = sum(dataset["length"])
        print(f"{split}: {total_tokens:,} tokens across {len(dataset):,} documents")

        # Write to flat uint16 binary file (matches nanoGPT approach)
        filename = output_dir / f"{split}.bin"
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        idx = 0
        for tokens in tqdm(dataset["tokens"], desc=f"Writing {split}"):
            arr[idx : idx + len(tokens)] = tokens
            idx += len(tokens)
        arr.flush()
        print(f"Saved {filename} ({filename.stat().st_size / 1e9:.2f} GB)")


def prepare_finetune(output_dir: Path, pretrain_dir: Path, num_proc: int = 4):
    """Prepare Alpaca instruction data with loss masks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/data/hf_cache/datasets")

    print("Loading yahma/alpaca-cleaned...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train", cache_dir=cache_dir)

    # Anti-forgetting: load 2500 samples from pretrain data
    print("Sampling 2500 FineWeb docs for anti-forgetting...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train[:5000]", cache_dir=cache_dir)
    fw_sample = fw.shuffle(seed=42).select(range(2500))

    def format_alpaca(example: dict) -> dict:
        if example.get("input", "").strip():
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        response = example["output"]
        return {"prompt": prompt, "response": response}

    def tokenize_instruction(example: dict) -> dict:
        prompt_tokens = TOKENIZER.encode_ordinary(example["prompt"])
        response_tokens = TOKENIZER.encode_ordinary(example["response"]) + [EOT]
        tokens = prompt_tokens + response_tokens
        # loss_mask: 0 for prompt (ignored), 1 for response (trained)
        mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
        return {"tokens": tokens, "loss_mask": mask, "length": len(tokens)}

    def tokenize_pretrain_doc(example: dict) -> dict:
        tokens = tokenize_text(example["text"])
        mask = [1] * len(tokens)  # train on everything
        return {"tokens": tokens, "loss_mask": mask, "length": len(tokens)}

    # Process instruction data
    ds_formatted = ds.map(format_alpaca)
    ds_tokenized = ds_formatted.map(tokenize_instruction, num_proc=num_proc, remove_columns=ds_formatted.column_names)

    # Process anti-forgetting data
    fw_tokenized = fw_sample.map(tokenize_pretrain_doc, num_proc=num_proc, remove_columns=fw_sample.column_names)

    # Combine
    from datasets import concatenate_datasets
    combined = concatenate_datasets([ds_tokenized, fw_tokenized]).shuffle(seed=42)
    combined = combined.train_test_split(test_size=0.02, seed=42)

    # Save as JSONL (variable length — can't easily use memmap for finetuning)
    for split, dataset in combined.items():
        filename = output_dir / f"{split}.jsonl"
        with open(filename, "w") as f:
            for example in tqdm(dataset, desc=f"Writing {split}"):
                f.write(json.dumps({"tokens": example["tokens"], "loss_mask": example["loss_mask"]}) + "\n")
        print(f"Saved {filename} ({len(dataset):,} examples)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pretrain", "finetune", "all"], default="all")
    parser.add_argument("--output_dir", type=Path, default=Path("data"))
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    if args.dataset in ("pretrain", "all"):
        prepare_pretrain(args.output_dir / "pretrain", num_proc=args.num_proc)

    if args.dataset in ("finetune", "all"):
        prepare_finetune(args.output_dir / "finetune", args.output_dir / "pretrain", num_proc=args.num_proc)
