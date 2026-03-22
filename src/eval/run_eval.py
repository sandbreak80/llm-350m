"""
Benchmark evaluation for pretrained and finetuned LLM checkpoints.

Runs HellaSwag, LAMBADA, ARC-Easy, ARC-Challenge, and WinoGrande on CPU,
logs results to W&B under a dedicated "benchmark-evals" run.

Usage:
    # Pretrain checkpoint (HellaSwag + LAMBADA only, fast)
    python src/eval/run_eval.py --checkpoint checkpoints/pretrain/ckpt_0040000.pt

    # Finetune checkpoint (full suite)
    python src/eval/run_eval.py --checkpoint checkpoints/finetune/best.pt --full --wandb_project llm-350m-finetune
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import tiktoken

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.model.model import LLM

DEVICE = torch.device("cpu")  # CPU only — training occupies GPU full-time
ENC = tiktoken.get_encoding("gpt2")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path) -> tuple[LLM, int, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LLM(ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["iter"], ckpt.get("val_loss", float("nan"))


# ── Scoring primitives ────────────────────────────────────────────────────────

@torch.no_grad()
def loglikelihood(model: LLM, context: str, continuation: str) -> float:
    """Sum of log-probs of continuation tokens given context."""
    ctx_tokens = ENC.encode(context)
    cont_tokens = ENC.encode(continuation)
    if not cont_tokens:
        return 0.0

    all_tokens = ctx_tokens + cont_tokens
    block_size = model.config.block_size

    # Truncate from the left if too long, keeping all continuation tokens
    if len(all_tokens) > block_size:
        all_tokens = all_tokens[-(block_size):]
        cont_start = len(all_tokens) - len(cont_tokens)
    else:
        cont_start = len(ctx_tokens)

    tokens = torch.tensor([all_tokens], dtype=torch.long)
    logits, _ = model(tokens)

    # Positions [cont_start-1 .. -1] predict continuation tokens
    shift_logits = logits[0, cont_start - 1: cont_start - 1 + len(cont_tokens)]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    cont_tensor = torch.tensor(cont_tokens)
    return log_probs[range(len(cont_tokens)), cont_tensor].sum().item()


# ── HellaSwag ─────────────────────────────────────────────────────────────────

def eval_hellaswag(model: LLM, num_samples: int = 500) -> float:
    """Multiple-choice commonsense completion. Score = acc on validation set."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)

    correct, total = 0, 0
    for example in ds:
        if total >= num_samples:
            break
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        scores = [loglikelihood(model, ctx, " " + e) for e in endings]
        if scores.index(max(scores)) == label:
            correct += 1
        total += 1

        if total % 100 == 0:
            print(f"  HellaSwag {total}/{num_samples}  acc={correct/total:.3f}", flush=True)

    return correct / total if total else 0.0


# ── LAMBADA ───────────────────────────────────────────────────────────────────

def eval_lambada(model: LLM, num_samples: int = 1000) -> float:
    """Last-word prediction accuracy. Checks if argmax next-token == first token of last word."""
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", split="test", streaming=True)

    correct, total = 0, 0
    for example in ds:
        if total >= num_samples:
            break
        text = example["text"]
        parts = text.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        context, last_word = parts

        last_word_tokens = ENC.encode(" " + last_word)
        if not last_word_tokens:
            continue

        ctx_tokens = ENC.encode(context)
        block_size = model.config.block_size
        if len(ctx_tokens) >= block_size:
            ctx_tokens = ctx_tokens[-(block_size - 1):]

        tokens = torch.tensor([ctx_tokens], dtype=torch.long)
        with torch.no_grad():
            logits, _ = model(tokens)
        pred = logits[0, -1].argmax().item()

        if pred == last_word_tokens[0]:
            correct += 1
        total += 1

        if total % 250 == 0:
            print(f"  LAMBADA {total}/{num_samples}  acc={correct/total:.3f}", flush=True)

    return correct / total if total else 0.0


# ── ARC ───────────────────────────────────────────────────────────────────────

def eval_arc(model: LLM, subset: str = "ARC-Easy", num_samples: int = 500) -> float:
    """ARC multiple-choice science QA. subset = 'ARC-Easy' or 'ARC-Challenge'."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", subset, split="validation", streaming=True)

    correct, total = 0, 0
    for example in ds:
        if total >= num_samples:
            break
        question = example["question"]
        choices = example["choices"]["text"]
        labels = example["choices"]["label"]
        answer_key = example["answerKey"]

        scores = [loglikelihood(model, question, " " + c) for c in choices]
        pred_label = labels[scores.index(max(scores))]
        if pred_label == answer_key:
            correct += 1
        total += 1

        if total % 100 == 0:
            print(f"  ARC-{subset.split('-')[1]} {total}/{num_samples}  acc={correct/total:.3f}", flush=True)

    return correct / total if total else 0.0


# ── WinoGrande ────────────────────────────────────────────────────────────────

def eval_winogrande(model: LLM, num_samples: int = 500) -> float:
    """WinoGrande commonsense pronoun resolution. Fill-in-the-blank, 2-choice."""
    from datasets import load_dataset
    ds = load_dataset("winogrande", "winogrande_xl", split="validation", streaming=True)

    correct, total = 0, 0
    for example in ds:
        if total >= num_samples:
            break
        sentence = example["sentence"]
        option1 = example["option1"]
        option2 = example["option2"]
        answer = example["answer"]  # "1" or "2"

        parts = sentence.split("_")
        if len(parts) != 2:
            continue
        context, suffix = parts[0], parts[1]

        score1 = loglikelihood(model, context, option1 + suffix)
        score2 = loglikelihood(model, context, option2 + suffix)
        pred = "1" if score1 > score2 else "2"
        if pred == answer:
            correct += 1
        total += 1

        if total % 100 == 0:
            print(f"  WinoGrande {total}/{num_samples}  acc={correct/total:.3f}", flush=True)

    return correct / total if total else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hellaswag_samples", type=int, default=500)
    parser.add_argument("--lambada_samples", type=int, default=1000)
    parser.add_argument("--arc_samples", type=int, default=500)
    parser.add_argument("--winogrande_samples", type=int, default=500)
    parser.add_argument("--full", action="store_true", help="Run full suite: HellaSwag+LAMBADA+ARC+WinoGrande")
    parser.add_argument("--wandb_project", type=str, default="llm-350m-pretrain")
    parser.add_argument("--wandb_entity", type=str, default="bstoner-riffyx")
    args = parser.parse_args()

    print(f"\n=== Benchmark Eval: {args.checkpoint.name} ===")
    model, iteration, val_loss = load_model(args.checkpoint)
    print(f"Iter: {iteration:,}  |  val_loss: {val_loss:.4f}  |  device: CPU")
    print(f"Model params: {model.num_params():,}")

    results = {"eval/val_loss": val_loss}

    print(f"\nRunning HellaSwag ({args.hellaswag_samples} samples)...")
    hellaswag_acc = eval_hellaswag(model, args.hellaswag_samples)
    results["eval/hellaswag_acc"] = hellaswag_acc
    print(f"  → HellaSwag acc: {hellaswag_acc*100:.2f}%")

    print(f"\nRunning LAMBADA ({args.lambada_samples} samples)...")
    lambada_acc = eval_lambada(model, args.lambada_samples)
    results["eval/lambada_acc"] = lambada_acc
    print(f"  → LAMBADA acc: {lambada_acc*100:.2f}%")

    if args.full:
        print(f"\nRunning ARC-Easy ({args.arc_samples} samples)...")
        arc_easy_acc = eval_arc(model, "ARC-Easy", args.arc_samples)
        results["eval/arc_easy_acc"] = arc_easy_acc
        print(f"  → ARC-Easy acc: {arc_easy_acc*100:.2f}%")

        print(f"\nRunning ARC-Challenge ({args.arc_samples} samples)...")
        arc_challenge_acc = eval_arc(model, "ARC-Challenge", args.arc_samples)
        results["eval/arc_challenge_acc"] = arc_challenge_acc
        print(f"  → ARC-Challenge acc: {arc_challenge_acc*100:.2f}%")

        print(f"\nRunning WinoGrande ({args.winogrande_samples} samples)...")
        winogrande_acc = eval_winogrande(model, args.winogrande_samples)
        results["eval/winogrande_acc"] = winogrande_acc
        print(f"  → WinoGrande acc: {winogrande_acc*100:.2f}%")

    # Summary
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY — {args.checkpoint.name} (iter {iteration:,})")
    print(f"{'='*50}")
    for k, v in results.items():
        label = k.replace("eval/", "").replace("_acc", "").replace("_", " ").title()
        if "acc" in k:
            print(f"  {label:<20} {v*100:.2f}%")
        else:
            print(f"  {label:<20} {v:.4f}")
    print(f"{'='*50}")

    # Log to W&B
    try:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            id="benchmark-evals",
            name="benchmark-evals",
            resume="allow",
        )
        wandb.log(results, step=iteration)
        wandb.finish()
        print(f"\nLogged to W&B (step={iteration})")
    except Exception as e:
        print(f"\nW&B logging failed: {e}")
        print(f"Results: {results}")

    print("\nDone.")


if __name__ == "__main__":
    main()
