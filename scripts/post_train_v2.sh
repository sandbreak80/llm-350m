#!/bin/bash
# post_train_v2.sh — Full post-training pipeline for V2
#
# Runs automatically after finetuning completes:
#   1. Full benchmark eval (HellaSwag, LAMBADA, ARC, WinoGrande)
#   2. Sample generation
#   3. Export to HuggingFace as sandbreak80sd/llm-350m-instruct-v2
#   4. Build llama.cpp + convert to GGUF (f16, q4_k_m, q8_0)
#   5. Upload GGUFs + Modelfile to HF
#   6. Generate and push model card README
#   7. Write completion marker to S3

set -euo pipefail

PROJ_DIR="/home/ec2-user/llm-project"
PYTHON="/opt/pytorch/bin/python"
PIP="/opt/pytorch/bin/pip"
CHECKPOINT="$PROJ_DIR/checkpoints/finetune/best.pt"
HF_REPO="sandbreak80sd/llm-350m-instruct-v2"
HF_EXPORT_DIR="$PROJ_DIR/hf_export_v2"
GGUF_DIR="$PROJ_DIR/gguf_v2"
LOG="/tmp/post_train_v2.log"
RESULTS_JSON="/tmp/v2_eval_results.json"
S3_BUCKET="bstoner-llm-checkpoints-536277006919"

exec > >(tee -a "$LOG") 2>&1

cd "$PROJ_DIR"

echo "============================================"
echo "  POST-TRAIN V2 PIPELINE"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================"

# ── Step 1: Full Benchmark Eval ───────────────────────────────────────────────
echo ""
echo "=== STEP 1: Benchmark Eval ==="
$PYTHON src/eval/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --full \
    --wandb_project llm-350m-finetune-v2 \
    2>&1 | tee /tmp/v2_eval.log

# Parse results from eval output into JSON
$PYTHON - << 'PYEOF'
import re, json

with open("/tmp/v2_eval.log") as f:
    text = f.read()

results = {}
patterns = {
    "hellaswag": r"HellaSwag\s+(\d+\.\d+)%",
    "lambada":   r"Lambada\s+(\d+\.\d+)%",
    "arc_easy":  r"Arc Easy\s+(\d+\.\d+)%",
    "arc_challenge": r"Arc Challenge\s+(\d+\.\d+)%",
    "winogrande": r"Winogrande\s+(\d+\.\d+)%",
    "val_loss":  r"Val Loss\s+(\d+\.\d+)",
}
for key, pat in patterns.items():
    m = re.search(pat, text)
    results[key] = float(m.group(1)) if m else None

with open("/tmp/v2_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Parsed results:", results)
PYEOF

echo "Eval complete."

# ── Step 2: Sample Generation ─────────────────────────────────────────────────
echo ""
echo "=== STEP 2: Sample Generation ==="
$PYTHON scripts/generate.py \
    --checkpoint "$CHECKPOINT" \
    --samples \
    --format chatml \
    2>&1 | tee /tmp/v2_samples.log
echo "Generation complete."

# ── Step 3: Export to HuggingFace ─────────────────────────────────────────────
echo ""
echo "=== STEP 3: Export to HuggingFace ($HF_REPO) ==="
mkdir -p "$HF_EXPORT_DIR"
$PYTHON scripts/export_to_hf.py \
    --checkpoint "$CHECKPOINT" \
    --repo_id "$HF_REPO" \
    --output_dir "$HF_EXPORT_DIR" \
    --push \
    2>&1
echo "HF export complete."

# ── Step 4: Build llama.cpp ───────────────────────────────────────────────────
echo ""
echo "=== STEP 4: Build llama.cpp ==="
if [ ! -f "$HOME/llama.cpp/build/bin/llama-quantize" ]; then
    cd "$HOME"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF 2>&1
    cmake --build build --config Release -j$(nproc) --target llama-quantize 2>&1
    # Install python deps for conversion
    $PIP install -q -r requirements.txt 2>/dev/null || true
    $PIP install -q gguf sentencepiece 2>/dev/null || true
    cd "$PROJ_DIR"
    echo "llama.cpp built."
else
    echo "llama.cpp already built."
fi

# ── Step 5: GGUF Conversion ───────────────────────────────────────────────────
echo ""
echo "=== STEP 5: GGUF Conversion ==="
mkdir -p "$GGUF_DIR"
cd "$HOME/llama.cpp"

# Convert HF export to F16 GGUF
python3 convert_hf_to_gguf.py \
    "$HF_EXPORT_DIR" \
    --outfile "$GGUF_DIR/llm-350m-instruct-v2-f16.gguf" \
    --outtype f16 2>&1

# Quantize Q4_K_M
build/bin/llama-quantize \
    "$GGUF_DIR/llm-350m-instruct-v2-f16.gguf" \
    "$GGUF_DIR/llm-350m-instruct-v2-q4_k_m.gguf" \
    Q4_K_M 2>&1

# Quantize Q8_0
build/bin/llama-quantize \
    "$GGUF_DIR/llm-350m-instruct-v2-f16.gguf" \
    "$GGUF_DIR/llm-350m-instruct-v2-q8_0.gguf" \
    Q8_0 2>&1

echo "GGUF files:"
ls -lh "$GGUF_DIR/"
cd "$PROJ_DIR"

# ── Step 6: Generate Model Card README ───────────────────────────────────────
echo ""
echo "=== STEP 6: Generate Model Card ==="
$PYTHON - << 'PYEOF'
import json

with open("/tmp/v2_eval_results.json") as f:
    r = json.load(f)

with open("/tmp/v2_samples.log") as f:
    samples_text = f.read()

def fmt(v):
    return f"{v:.2f}%" if v is not None else "TBD"

readme = f"""---
language:
- en
license: apache-2.0
tags:
- text-generation
- causal-lm
- llama
- gqa
- rope
- swiglu
- from-scratch
- pretraining
- instruction-tuned
- chatml
datasets:
- HuggingFaceFW/fineweb-edu
- teknium/OpenHermes-2.5
metrics:
- perplexity
pipeline_tag: text-generation
---

# LLM-350M-Instruct-V2

This is V2 of a 350M parameter language model trained entirely from scratch as a personal learning project. V2 improves on [V1](https://huggingface.co/sandbreak80sd/llm-350m-instruct) by replacing the Alpaca-cleaned finetuning dataset with OpenHermes-2.5 — 200K GPT-4 generated examples in ChatML format instead of 52K GPT-3.5 examples in Alpaca format.

I'm not a researcher. I don't work at a big lab. I just wanted to understand how LLMs actually work by building one. Everything is documented — nothing is hand-wavy.

**What changed from V1:** Better finetuning data (GPT-4 vs GPT-3.5), ChatML format (vs Alpaca), lower learning rate (1e-5 vs 2e-5), 4000 iters (vs 1500). Same pretrained base.

**[V1 model](https://huggingface.co/sandbreak80sd/llm-350m-instruct)** | **[Training code](https://github.com/sandbreak80/llm-350m)** | **[W&B](https://wandb.ai/bstoner-riffyx/llm-350m-finetune-v2)**

---

## Benchmarks

| Benchmark | V1 (Alpaca) | V2 (OpenHermes) | Δ |
|---|---|---|---|
| HellaSwag | 38.40% | {fmt(r.get('hellaswag'))} | {"+" if r.get('hellaswag') and r['hellaswag'] > 38.40 else ""}{f"{r['hellaswag']-38.40:.2f}%" if r.get('hellaswag') else "TBD"} |
| LAMBADA | 34.00% | {fmt(r.get('lambada'))} | {"+" if r.get('lambada') and r['lambada'] > 34.00 else ""}{f"{r['lambada']-34.00:.2f}%" if r.get('lambada') else "TBD"} |
| ARC-Easy | 58.20% | {fmt(r.get('arc_easy'))} | {"+" if r.get('arc_easy') and r['arc_easy'] > 58.20 else ""}{f"{r['arc_easy']-58.20:.2f}%" if r.get('arc_easy') else "TBD"} |
| ARC-Challenge | 27.76% | {fmt(r.get('arc_challenge'))} | {"+" if r.get('arc_challenge') and r['arc_challenge'] > 27.76 else ""}{f"{r['arc_challenge']-27.76:.2f}%" if r.get('arc_challenge') else "TBD"} |
| WinoGrande | 52.80% | {fmt(r.get('winogrande'))} | {"+" if r.get('winogrande') and r['winogrande'] > 52.80 else ""}{f"{r['winogrande']-52.80:.2f}%" if r.get('winogrande') else "TBD"} |

Val loss: {r.get('val_loss', 'TBD')} (V1: 1.7189)

---

## What's different from V1

| | V1 | V2 |
|---|---|---|
| Finetune dataset | yahma/alpaca-cleaned (52K, GPT-3.5) | teknium/OpenHermes-2.5 (200K, GPT-4) |
| Prompt format | Alpaca | **ChatML** |
| Learning rate | 2e-5 | 1e-5 |
| Finetune iters | 1,500 | 4,000 |
| Anti-forgetting | 2,500 FineWeb samples | 10,000 FineWeb samples |

## Model Architecture

Same as V1 — modern LLaMA-style improvements over GPT-2 at 350M parameters:
RoPE positional encoding, RMSNorm (pre-norm), SwiGLU activations, Grouped Query Attention (4 KV heads / 16 query heads), 2048 token context, Flash Attention 2, tied embeddings.

Full architecture details in the [V1 model card](https://huggingface.co/sandbreak80sd/llm-350m-instruct).

## Quick Start

### Ollama
```bash
ollama pull sandbreak80sd/llm-350m-instruct-v2
ollama run sandbreak80sd/llm-350m-instruct-v2
```

### llama.cpp / LM Studio
Download `llm-350m-instruct-v2-q4_k_m.gguf` from the Files tab.

### Python (HuggingFace Transformers)
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("sandbreak80sd/llm-350m-instruct-v2")
model = AutoModelForCausalLM.from_pretrained("sandbreak80sd/llm-350m-instruct-v2", torch_dtype=torch.bfloat16)

# ChatML format
prompt = "<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\nExplain what a neural network is.<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Limitations

Same fundamental limitations as V1 — this is a 350M parameter model trained on a hobbyist budget:
- **Code generation**: May produce structurally plausible but semantically wrong code
- **Math**: Unreliable beyond simple arithmetic
- **Knowledge cutoff**: FineWeb-Edu training data, no recent events
- **No safety alignment**: Not suitable for production use

## Training Cost

~$10 for V2 finetuning on a single g6e.xlarge (L4 GPU). Full project cost including pretraining: ~$300.

## Citation

```bibtex
@misc{{llm-350m-instruct-v2,
  author = {{Stoner, Brad}},
  title  = {{LLM-350M-Instruct-V2: A 350M parameter LLM trained from scratch}},
  year   = {{2026}},
  url    = {{https://huggingface.co/sandbreak80sd/llm-350m-instruct-v2}},
  note   = {{Training code: https://github.com/sandbreak80/llm-350m}}
}}
```
"""

with open("/tmp/v2_readme.md", "w") as f:
    f.write(readme)

print("README generated.")
PYEOF

# Copy README to HF export dir and push
cp /tmp/v2_readme.md "$HF_EXPORT_DIR/README.md"

$PYTHON - << 'PYEOF'
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="/tmp/v2_readme.md",
    path_in_repo="README.md",
    repo_id="sandbreak80sd/llm-350m-instruct-v2",
    repo_type="model",
    commit_message="Add model card with benchmark results",
)
print("README pushed to HF.")
PYEOF

# ── Step 7: Upload GGUFs + Modelfile to HF ───────────────────────────────────
echo ""
echo "=== STEP 7: Upload GGUFs to HF ==="
$PYTHON - << 'PYEOF'
from huggingface_hub import HfApi
import os

api = HfApi()
gguf_dir = "/home/ec2-user/llm-project/gguf_v2"

for fname in ["llm-350m-instruct-v2-f16.gguf", "llm-350m-instruct-v2-q4_k_m.gguf", "llm-350m-instruct-v2-q8_0.gguf"]:
    fpath = os.path.join(gguf_dir, fname)
    if os.path.exists(fpath):
        print(f"Uploading {fname}...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id="sandbreak80sd/llm-350m-instruct-v2",
            repo_type="model",
            commit_message=f"Add {fname}",
        )
        print(f"  Done: {fname}")

# Upload Modelfile
api.upload_file(
    path_or_fileobj="/home/ec2-user/llm-project/Modelfile",
    path_in_repo="Modelfile",
    repo_id="sandbreak80sd/llm-350m-instruct-v2",
    repo_type="model",
    commit_message="Add Ollama Modelfile (ChatML template)",
)
print("All files uploaded.")
PYEOF

# ── Step 8: Save V2 checkpoint to S3 ─────────────────────────────────────────
echo ""
echo "=== STEP 8: Save V2 checkpoint to S3 ==="
aws s3 cp "$CHECKPOINT" "s3://$S3_BUCKET/checkpoints/finetune_v2/best.pt"
echo "V2 checkpoint saved to S3."

# ── Step 9: Write completion marker ──────────────────────────────────────────
echo ""
echo "=== PIPELINE COMPLETE ==="
date -u '+%Y-%m-%d %H:%M:%S UTC'
cat /tmp/v2_eval_results.json

# Upload full log + completion marker to S3
aws s3 cp "$LOG" "s3://$S3_BUCKET/v2_pipeline.log"
echo "done" | aws s3 cp - "s3://$S3_BUCKET/v2_pipeline_complete.txt"
echo "Completion marker written to S3."
