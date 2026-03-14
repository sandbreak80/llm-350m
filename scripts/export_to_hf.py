"""
Export trained model to HuggingFace format.

Converts our custom LLM to a format compatible with the HuggingFace
transformers library (AutoModelForCausalLM), then pushes to the Hub.

Usage:
    python scripts/export_to_hf.py \
        --checkpoint checkpoints/finetune/best.pt \
        --repo_id YOUR_HF_USERNAME/your-model-name \
        --push
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo
import tiktoken

from src.model.config import ModelConfig
from src.model.model import LLM


# HuggingFace config.json for AutoModelForCausalLM compatibility
# Maps our architecture to a known transformers config format.
# We use a custom "llama"-adjacent config since our arch is similar.
def build_hf_config(model_cfg: ModelConfig) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],  # closest HF architecture to ours
        "model_type": "llama",
        "hidden_size": model_cfg.n_embd,
        "intermediate_size": model_cfg.intermediate_size,
        "num_hidden_layers": model_cfg.n_layers,
        "num_attention_heads": model_cfg.n_heads,
        "num_key_value_heads": model_cfg.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": model_cfg.block_size,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,
        "rope_theta": model_cfg.rope_theta,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
        "vocab_size": model_cfg.vocab_size,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
    }


def remap_weights(state_dict: dict, model_cfg: ModelConfig) -> dict:
    """Remap our weight names to HuggingFace LlamaForCausalLM names."""
    mapping = {}
    mapping["model.embed_tokens.weight"] = state_dict["token_emb.weight"]
    mapping["model.norm.weight"] = state_dict["norm.weight"]
    mapping["lm_head.weight"] = state_dict["token_emb.weight"]  # tied

    for i in range(model_cfg.n_layers):
        prefix = f"layers.{i}"
        hf_prefix = f"model.layers.{i}"

        mapping[f"{hf_prefix}.input_layernorm.weight"] = state_dict[f"{prefix}.attn_norm.weight"]
        mapping[f"{hf_prefix}.post_attention_layernorm.weight"] = state_dict[f"{prefix}.ffn_norm.weight"]

        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = state_dict[f"{prefix}.attn.q_proj.weight"]
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = state_dict[f"{prefix}.attn.k_proj.weight"]
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = state_dict[f"{prefix}.attn.v_proj.weight"]
        mapping[f"{hf_prefix}.self_attn.o_proj.weight"] = state_dict[f"{prefix}.attn.o_proj.weight"]

        mapping[f"{hf_prefix}.mlp.gate_proj.weight"] = state_dict[f"{prefix}.ffn.gate_proj.weight"]
        mapping[f"{hf_prefix}.mlp.up_proj.weight"] = state_dict[f"{prefix}.ffn.up_proj.weight"]
        mapping[f"{hf_prefix}.mlp.down_proj.weight"] = state_dict[f"{prefix}.ffn.down_proj.weight"]

    return mapping


def export(checkpoint_path: str, output_dir: str, repo_id: str, push: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg: ModelConfig = checkpoint["model_config"]

    print(f"Loaded checkpoint: val_loss={checkpoint.get('val_loss', 'N/A')}, iter={checkpoint.get('iter', 'N/A')}")

    # Remap weights to HF format
    state_dict = checkpoint["model"]
    hf_weights = remap_weights(state_dict, model_cfg)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    save_file(hf_weights, output_path / "model.safetensors")
    print(f"Saved weights: {output_path / 'model.safetensors'}")

    # Save HF config.json
    hf_config = build_hf_config(model_cfg)
    with open(output_path / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)
    print(f"Saved config: {output_path / 'config.json'}")

    # Save tokenizer files (GPT-2 tokenizer)
    enc = tiktoken.get_encoding("gpt2")
    tokenizer_config = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "model_max_length": model_cfg.block_size,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open(output_path / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Copy GPT-2 tokenizer vocab files
    from transformers import GPT2Tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.save_pretrained(str(output_path))
    print(f"Saved tokenizer files")

    print(f"\nModel ready at: {output_path}")
    print(f"Parameter count: {sum(t.numel() for t in hf_weights.values() if 'embed' not in t):,} (non-embedding)")

    if push:
        print(f"\nPushing to HuggingFace Hub: {repo_id}")
        api = HfApi()
        create_repo(repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_path),
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="hf_export")
    parser.add_argument("--repo_id", required=True, help="e.g. YourUsername/your-model-350M")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub")
    args = parser.parse_args()

    export(args.checkpoint, args.output_dir, args.repo_id, args.push)
