"""
Generate text from a finetuned LLM checkpoint.

Supports Alpaca format (v1) and ChatML format (v2/OpenHermes).

Usage:
    # Single prompt (ChatML, default)
    python scripts/generate.py --checkpoint checkpoints/finetune/best.pt \
        --instruction "Explain the difference between supervised and unsupervised learning."

    # Alpaca format (v1 model)
    python scripts/generate.py --checkpoint checkpoints/finetune/best.pt \
        --format alpaca --instruction "Explain supervised learning."

    # Interactive mode
    python scripts/generate.py --checkpoint checkpoints/finetune/best.pt

    # Run all built-in sample prompts for model card examples
    python scripts/generate.py --checkpoint checkpoints/finetune/best.pt --samples
"""

import argparse
import sys
from pathlib import Path

import torch
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model.model import LLM

ENC = tiktoken.get_encoding("gpt2")

SAMPLE_PROMPTS = [
    {
        "instruction": "Explain the difference between supervised and unsupervised learning.",
        "input": "",
    },
    {
        "instruction": "Write a Python function that checks if a string is a palindrome.",
        "input": "",
    },
    {
        "instruction": "What causes the seasons to change on Earth?",
        "input": "",
    },
    {
        "instruction": "Summarize the following passage in one sentence.",
        "input": "The mitochondria is often referred to as the powerhouse of the cell. It is a double-membrane-bound organelle found in the cytoplasm of eukaryotic cells. Mitochondria generate most of the cell's supply of ATP, which is used as a source of chemical energy.",
    },
    {
        "instruction": "Give three tips for staying focused while working from home.",
        "input": "",
    },
]


def alpaca_prompt(instruction: str, input_text: str = "") -> str:
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def chatml_prompt(instruction: str, input_text: str = "", system: str = "You are a helpful assistant.") -> str:
    user_content = f"{instruction}\n\n{input_text}".strip() if input_text.strip() else instruction
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


IM_END = "<|im_end|>"


@torch.no_grad()
def generate(
    model: LLM,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    tokens = ENC.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits, _ = model(x)
        logits = logits[0, -1, :] / temperature

        # top-k filter
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[-1]] = float("-inf")

        # top-p (nucleus) filter
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        sorted_probs[cumsum - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum()

        next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
        x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == ENC.eot_token:
            break

    new_tokens = x[0, len(tokens):].tolist()
    # Strip at EOS
    if ENC.eot_token in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(ENC.eot_token)]
    text = ENC.decode(new_tokens)
    # Strip at <|im_end|> (ChatML stop token)
    if IM_END in text:
        text = text[:text.index(IM_END)]
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--samples", action="store_true", help="Run all built-in sample prompts")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    parser.add_argument("--format", choices=["chatml", "alpaca"], default="chatml",
                        help="Prompt format: chatml (v2, default) or alpaca (v1)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LLM(ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded (val_loss={ckpt.get('val_loss', 'N/A'):.4f}, iter={ckpt.get('iter', 'N/A'):,}, device={device})\n")

    build_prompt = chatml_prompt if args.format == "chatml" else alpaca_prompt
    print(f"Prompt format: {args.format}\n")

    if args.samples:
        for i, p in enumerate(SAMPLE_PROMPTS, 1):
            prompt = build_prompt(p["instruction"], p["input"])
            print(f"{'='*60}")
            print(f"[{i}/{len(SAMPLE_PROMPTS)}] Instruction: {p['instruction']}")
            if p["input"]:
                print(f"Input: {p['input'][:80]}...")
            print(f"{'='*60}")
            output = generate(model, prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k, device)
            print(f"Response:\n{output}\n")

    elif args.instruction:
        prompt = build_prompt(args.instruction, args.input)
        output = generate(model, prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k, device)
        print(f"Response:\n{output}")

    else:
        print("Interactive mode. Type 'quit' to exit.\n")
        while True:
            instruction = input("Instruction: ").strip()
            if instruction.lower() in ("quit", "exit", "q"):
                break
            input_text = input("Input (optional, press Enter to skip): ").strip()
            prompt = build_prompt(instruction, input_text)
            output = generate(model, prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k, device)
            print(f"\nResponse:\n{output}\n")


if __name__ == "__main__":
    main()
