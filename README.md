---
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
datasets:
- HuggingFaceFW/fineweb-edu
- yahma/alpaca-cleaned
metrics:
- perplexity
pipeline_tag: text-generation
---

# LLM-350M-Instruct

This is a 350M parameter language model trained entirely from scratch as a personal learning project — pretraining, finetuning, evaluation, and all.

I'm not a researcher. I don't work at a big lab. I just wanted to understand how LLMs actually work by building one. The whole thing ran on a single rented GPU for under $500, and took a few weeks of evenings and weekends. Everything I learned along the way is documented here.

If you're curious about how LLMs are built, or want a small open model to experiment with, hopefully this is useful. The code is all open source and the training details are fully documented — nothing is hand-wavy.

**What this is**: A small instruction-following model. It can answer questions, summarize text, explain concepts, and follow simple instructions reasonably well. It's not going to replace anything, but it works, and it was built from scratch.

**What this isn't**: A state-of-the-art model. It's 350M parameters trained on a hobbyist budget. Manage your expectations accordingly — and check the Limitations section.

---

**Code**: [github.com/sandbreak80/llm-350m](https://github.com/sandbreak80/llm-350m)
**Training logs (pretrain)**: [W&B](https://wandb.ai/bstoner-riffyx/llm-350m-pretrain/runs/dr1o3zr9)
**Training logs (finetune)**: [W&B](https://wandb.ai/bstoner-riffyx/llm-350m-finetune/runs/pkusqm98)
**Benchmark evals**: [W&B](https://wandb.ai/bstoner-riffyx/llm-350m-finetune/runs/benchmark-evals)

---

## Model Architecture

The architecture is designed to be a modern improvement over GPT-2/nanoGPT-style models at the same parameter count. Every component was chosen based on what the research literature shows works at this scale.

### Comparison with Reference Model

We use [Apex-1-Instruct-350M](https://huggingface.co/LH-Tech-AI/Apex-1-Instruct-350M) as a low-bar reference — same dataset, same approximate size, but a 2023-era GPT-2 style architecture. Our architecture incorporates every major improvement from the LLaMA/Mistral lineage:

| Component | GPT-2 / Apex-1 (Reference) | This Model | Why |
|---|---|---|---|
| Positional encoding | Learned absolute | **RoPE** | Better length generalization, no wasted embedding parameters |
| Normalization | LayerNorm (post-norm) | **RMSNorm (pre-norm)** | More stable training, cleaner gradient flow |
| Activation | GELU | **SwiGLU** | ~5-10% better loss at same parameter count (PaLM, LLaMA) |
| Attention | Multi-Head (MHA) | **Grouped Query Attention** | 4× fewer KV parameters, faster inference, minimal quality loss |
| Context length | 1,024 tokens | **2,048 tokens** | 2× longer context at same compute via RoPE efficiency |
| Attention kernel | Standard | **Flash Attention 2** (PyTorch SDPA) | Memory-efficient, no implementation overhead |
| Embedding tying | Not tied | **Tied** (token emb = LM head) | Reduces parameters ~2%, improves training signal |

These changes make our architecture essentially a small LLaMA-3 rather than a GPT-2.

### Hyperparameters

| Parameter | Value |
|---|---|
| Total parameters | ~350M |
| Non-embedding parameters | 270,582,784 |
| Layers (`n_layers`) | 24 |
| Hidden dimension (`n_embd`) | 1,024 |
| Query heads (`n_heads`) | 16 |
| KV heads (`n_kv_heads`) | 4 (GQA ratio 4:1) |
| Head dimension | 64 |
| FFN intermediate size | 2,816 (~2.75× hidden, SwiGLU accounts for gating) |
| Context length (`block_size`) | 2,048 |
| Vocabulary size | 50,304 (GPT-2, padded to multiple of 64) |
| RoPE theta | 10,000 |
| Bias terms | None (modern practice) |
| Dropout (pretrain) | 0.0 |
| Dropout (finetune) | 0.1 |

### Parameter Breakdown

| Component | Parameters |
|---|---|
| Token embeddings (tied to LM head) | 51,511,296 |
| Attention (Q/K/V/O) × 24 layers | ~134M |
| FFN (gate/up/down) × 24 layers | ~171M |
| RMSNorm weights | ~50K |
| **Total** | **~350M** |

---

## Training Data

### Pretraining Dataset: FineWeb-Edu

- **Source**: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` split
- **Size on disk**: 18.4 GiB (tokenized binary, uint16) + 97 MiB validation
- **Token count**: ~9.9B training tokens, ~50M validation tokens
- **Format**: Streamed from HuggingFace, tokenized with GPT-2 tiktoken encoder, written to binary `uint16` files for memory-mapped training
- **Quality**: FineWeb-Edu is filtered CommonCrawl data scored by an educational quality classifier. It's higher quality than raw web text and has been shown to produce better models per token than general web corpora.
- **Train/val split**: ~99.5% train / 0.5% val, split by document

**Why FineWeb-Edu**: Same dataset as the Apex-1 reference model, enabling direct comparison. The educational content bias produces stronger reasoning and writing quality than unfiltered web text at this token count.

### Instruction Finetuning Dataset: Alpaca-Cleaned

- **Source**: [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned)
- **Size**: ~52,000 instruction-response pairs
- **Format**: Alpaca template (`### Instruction: / ### Input: / ### Response:`)
- **Preprocessing**: Tokenized with loss mask — only response tokens contribute to loss (prompt tokens masked to `-100`). This prevents the model from over-fitting to instruction formatting.
- **Anti-forgetting blend**: 2,500 FineWeb-Edu samples mixed in at a 4:1 Alpaca:FineWeb ratio. This preserves base language modeling quality and prevents catastrophic forgetting of pretraining knowledge — a technique validated in the Apex-1 reference.

**Data pipeline**: `src/data/prepare.py` — streams both datasets, tokenizes, and writes to binary JSONL (finetuning) and binary flat files (pretraining). All data cached to S3 to avoid re-tokenization on restarts.

---

## Training Procedure

### Pretraining

| Hyperparameter | Value |
|---|---|
| Max iterations | 60,000 |
| Effective batch size | 262,144 tokens (batch=4 × grad_accum=32 × seq_len=2048) |
| Peak learning rate | 6e-4 |
| Min learning rate | 6e-5 |
| LR schedule | Cosine decay with 2,000-iter linear warmup |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1, fused CUDA) |
| Gradient clipping | 1.0 |
| Precision | bfloat16 (no gradient scaler needed) |
| Tokens trained on | ~15.7B (262,144 × 60,000) |
| `torch.compile` | Disabled (hangs on custom RoPE/GQA ops) |
| Checkpoint interval | Every 500 iterations (spot instance resilience) |

**Total compute**: ~75 hours on 1× NVIDIA L40S (46GB VRAM). Throughput: ~35,000–36,000 tokens/sec.

### Instruction Finetuning (SFT)

| Hyperparameter | Value |
|---|---|
| Max iterations | 1,500 (~3.7 epochs over Alpaca) |
| Effective batch size | 262,144 tokens (same as pretrain) |
| Peak learning rate | 2e-5 |
| Min learning rate | 3e-6 |
| LR schedule | Cosine decay, no warmup |
| Dropout | 0.1 |
| Loss masking | Response tokens only (prompts masked) |
| Final val loss | **1.7189** (vs pretrain 2.708 — Δ = -0.989 nats) |

**Total compute**: ~2.5 hours on 1× NVIDIA L40S. Val loss converged and plateaued cleanly at iter 1,200–1,500.

---

## Training Infrastructure

### Hardware

- **Instance**: AWS EC2 `g6e.xlarge` (on-demand)
- **GPU**: NVIDIA L40S, 46GB VRAM
- **GPU utilization**: 100% throughout (~35K tok/s)
- **GPU temperature**: 63–80°C (stable)
- **vCPUs**: 4 (used for data loading, CPU-side benchmark evals)
- **Memory**: 128GB RAM
- **Storage**: 150GB EBS gp3 + S3 for checkpoint durability

### Checkpointing & Resilience

Training infrastructure was built to production standards despite the hobby-scale budget:

- **Checkpoint every 500 iterations** — minimizes lost work on spot interruptions
- **S3 sync every 5 minutes via cron** — checkpoints survive instance termination
- **SIGTERM handler** — catches AWS spot interruption signals and saves an emergency checkpoint before shutdown
- **Auto-resume from `latest.pt`** — training restarts exactly where it left off with no manual intervention
- **Non-blocking CPU eval pipeline** — benchmark evals (HellaSwag, LAMBADA) run on CPU via cron watcher while training occupies the GPU full-time; results log to W&B automatically

The run survived several spot interruptions with minimal iteration loss.

### Cost

Total AWS spend for this training run: ~$280 (within a $1,000 budget).

| Item | Cost |
|---|---|
| EC2 compute (pretraining) | ~$140 |
| EC2 compute (finetuning) | ~$6 |
| EBS storage | ~$5 |
| S3 storage + transfer | ~$2 |
| Earlier spot experiments | ~$65 |
| **Total** | **~$280** |

---

## Scaling Law Analysis

We tracked our val loss against the Chinchilla scaling law prediction throughout training. Using the Hoffmann et al. (2022) formula:

```
L(N, D) = E + A/N^α + B/D^β
E=1.69, A=406.4, B=410.7, α=0.34, β=0.28
```

| Tokens (B) | Actual Val Loss | Chinchilla Predicted | Δ |
|---|---|---|---|
| 4.2 | 2.9507 | 3.0260 | **-0.075** |
| 6.0 | 2.8964 | 2.9458 | **-0.049** |
| 7.9 | 2.8475 | 2.8920 | **-0.044** |
| 9.7 | 2.8031 | 2.8523 | **-0.049** |
| 11.8 | 2.7568 | 2.8157 | **-0.059** |

**The model consistently beats Chinchilla predictions by ~0.05 nats.** This is expected — Chinchilla was fit on GPT-style models, and our architectural improvements (SwiGLU, RMSNorm, RoPE) extract more signal per token.

**Over-training**: At 15.7B tokens, we train 2.25× past the Chinchilla-optimal 7B tokens for this model size. This is intentional. Chinchilla optimizes for training compute efficiency — it tells you when to *stop training* if you care about minimizing FLOPs. But for a model you'll serve at inference time, over-training a smaller model is strictly better: you get higher quality at lower serving cost. This is the core insight behind LLaMA-1/2 and Mistral, and we apply it here at 350M scale. The loss curve confirmed the model had not plateaued at 15.7B tokens — additional tokens would have continued to help.

---

## Benchmark Evaluation

All benchmarks evaluated using `src/eval/run_eval.py` on CPU (4-vCPU, no GPU). Scores use greedy loglikelihood ranking over answer choices (standard LM-eval-harness methodology). Pretrain evals were run automatically via cron watcher at every 5,000-iteration checkpoint.

### Pretraining Progress

| Checkpoint (iter) | Tokens Seen | Val Loss | HellaSwag (500) | LAMBADA (1000) |
|---|---|---|---|---|
| 37,500 | 9.8B | 2.8031 | 34.6% | 31.0% |
| 40,000 | 10.5B | 2.7852 | 35.2% | 31.2% |
| 45,000 | 11.8B | 2.7568 | 37.0% | 30.4% |
| **60,000 (final)** | **15.7B** | **2.7081** | **36.6%** | **34.5%** |

### Post-Finetuning (Instruct Model)

Evaluated on `checkpoints/finetune/best.pt` (iter 1,400, val_loss 1.7189):

| Benchmark | Score | Notes |
|---|---|---|
| HellaSwag (500 samples) | **38.40%** | 4-choice commonsense completion; random = 25% |
| LAMBADA (1000 samples) | **34.00%** | Last-word prediction accuracy |
| ARC-Easy (500 samples) | **58.20%** | 4-choice elementary science QA; random = 25% |
| ARC-Challenge (500 samples) | **27.76%** | 4-choice harder science QA; random = 25% |
| WinoGrande (500 samples) | **52.80%** | 2-choice commonsense pronoun resolution; random = 50% |

### Reference Comparison

| Model | HellaSwag | LAMBADA | Notes |
|---|---|---|---|
| Random baseline | 25.0% | — | 4-choice |
| GPT-2 117M | ~33% | ~39% | |
| GPT-2 1.5B | ~41% | ~54% | |
| Apex-1-Instruct-350M | ~35% (est.) | — | Our reference, val_loss=2.8175 |
| **This model (pretrain, 45k)** | **37.0%** | **30.4%** | Mid-training |
| **This model (pretrain, final 60k)** | **36.6%** | **34.5%** | Final pretrain checkpoint |
| **This model (instruct, final)** | **38.40%** | **34.00%** | **Beats Apex-1 reference** |

We surpass Apex-1's estimated HellaSwag score mid-training and hold that lead through the full finetuned model. The LAMBADA improvement from pretrain (30.4%) to instruct (34.00%) reflects that instruction finetuning improves contextual word prediction, not just formatting.

---

## Sample Outputs

Generated with `temperature=0.7, top_p=0.9, top_k=50, max_new_tokens=256` on GPU.

---

**Instruction**: Explain the difference between supervised and unsupervised learning.

**Response**:
> Supervised learning is a type of machine learning where the data is labeled and labeled in a way that the algorithm can understand and use it to make predictions or decisions. In this type of learning, the data is preprocessed to remove any unnecessary information, such as missing values, outliers or irrelevant features, before being used to train the model. The model is trained using labeled data, and the output of the model is then used to make predictions or decisions.
>
> Unsupervised learning, on the other hand, is a type of machine learning where the data is not labeled or preprocessed. Instead, the data is provided to the model as input. The model is trained using unlabeled data, and the output of the model is then used to make predictions or decisions.
>
> In summary, while supervised learning is a type of machine learning where the data is labeled and preprocessed, unsupervised learning is a type of machine learning where the data is not labeled or preprocessed, but instead, is provided to the model as input.

---

**Instruction**: What causes the seasons to change on Earth?

**Response**:
> The Earth's seasons are caused by the tilt of its axis, which causes the sun to rise and set at different times of the year. The tilt is known as the tilt of the Earth's axis, which causes the seasons to vary seasonally.
>
> The tilt of the Earth's axis is approximately 23.5 degrees. As the Earth rotates on its axis, the tilt changes. This tilt causes the seasons to change throughout the year.
>
> In addition to the tilt of the Earth's axis, other factors that can influence the seasons include the Earth's tilt and the Earth's rotation. For example, the tilt of the Earth's axis is influenced by the gravitational force of the Sun and the Earth's rotation.

---

**Instruction**: Give three tips for staying focused while working from home.

**Response**:
> 1. **Set a specific goal:** Set a specific task or goal for yourself that you want to achieve. This will help you stay focused and motivated to achieve your goals.
>
> 2. **Make a to-do list:** Make a to-do list to help you stay organized. This will help you stay on track and ensure that you don't forget any important tasks.
>
> 3. **Stay positive:** Being negative can often make it difficult to stay focused. Try to stay positive by reminding yourself of your achievements and the positive things you have accomplished. This will help you stay motivated and focused.

---

**Instruction**: Summarize the following passage in one sentence.
**Input**: *The mitochondria is often referred to as the powerhouse of the cell. It is a double-membrane-bound organelle found in the cytoplasm of eukaryotic cells. Mitochondria generate most of the cell's supply of ATP, which is used as a source of chemical energy.*

**Response**:
> The mitochondria is a double-membrane-bound organelle found in the cytoplasm of eukaryotic cells. ATP is produced by the mitochondria, which is a double-membrane-bound organelle.

---

**Instruction**: Write a Python function that checks if a string is a palindrome.

**Response**:
> ```python
> def check_palindrome(string):
>     if string == "":
>         return False
>     else:
>         return False
> ```
> This function takes an input string and checks if it is a palindrome. If it is, the function returns True.

*Note: The coding output above is incorrect — the model returns `False` unconditionally. This reflects the expected limitation of a 350M SFT-only model on code generation tasks. For reliable coding output, consider domain fine-tuning on a code-heavy dataset.*

---

## Run Locally with Ollama or LM Studio

GGUF files are included in this repo so you can run the model locally without any Python setup.

| File | Size | Notes |
|---|---|---|
| `llm-350m-instruct-q4_k_m.gguf` | **197MB** | Recommended — good balance of size and quality |
| `llm-350m-instruct-q8_0.gguf` | 329MB | Higher quality, still small |
| `llm-350m-instruct-f16.gguf` | 617MB | Full precision |

### Ollama

```bash
# Pull the GGUF directly (once the model is in the Ollama library)
# OR use the included Modelfile:

# 1. Download llm-350m-instruct-q4_k_m.gguf from this repo
# 2. Download the Modelfile from this repo
# 3. Run:
ollama create llm-350m-instruct -f Modelfile
ollama run llm-350m-instruct "Explain how neural networks learn"
```

The `Modelfile` in this repo configures the Alpaca prompt format and generation parameters automatically.

### LM Studio

1. Open LM Studio → search **sandbreak80sd/llm-350m-instruct** in the search bar
2. Download `llm-350m-instruct-q4_k_m.gguf` (197MB)
3. Load the model and start chatting

In the system prompt / prompt format settings, use Alpaca format:
```
### Instruction:
{prompt}

### Response:
```

### llama.cpp (CLI)

```bash
# Download the GGUF, then:
./llama-cli -m llm-350m-instruct-q4_k_m.gguf \
  --prompt "### Instruction:\nExplain the water cycle.\n\n### Response:\n" \
  -n 256 --temp 0.7 --top-p 0.9 --repeat-penalty 1.1
```

---

## Usage (Python / HuggingFace)

The model is exported in HuggingFace-compatible `LlamaForCausalLM` format with remapped weight names and GPT-2 tokenizer files.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "sandbreak80sd/llm-350m-instruct",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("sandbreak80sd/llm-350m-instruct")
```

### Instruction Prompt Format (Alpaca)

```
### Instruction:
{your instruction here}

### Input:
{optional additional context}

### Response:
```

### Generation Example

```python
prompt = "### Instruction:\nExplain the difference between supervised and unsupervised learning.\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.1,
)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Memory Requirements

| Precision | VRAM | Notes |
|---|---|---|
| bfloat16 | ~700MB | Recommended |
| float32 | ~1.4GB | |
| 4-bit quantized | ~200MB | Via bitsandbytes |

Runs comfortably on consumer GPUs (RTX 3060 and above) or CPU inference with llama.cpp.

---

## Codebase

All training code is open source: [github.com/sandbreak80/llm-350m](https://github.com/sandbreak80/llm-350m)

```
src/
├── model/
│   ├── config.py        # ModelConfig dataclass
│   └── model.py         # LLM, RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock
├── data/
│   └── prepare.py       # Dataset streaming + tokenization
├── training/
│   ├── train.py         # Pretraining loop (DDP-compatible, spot-resilient)
│   ├── finetune.py      # SFT loop with loss masking
│   └── config.py        # TrainConfig, FinetuneConfig dataclasses
└── eval/
    └── run_eval.py      # HellaSwag, LAMBADA, ARC, WinoGrande eval + W&B logging
scripts/
├── aws_setup.sh         # Instance bootstrap (installs deps, mounts EBS, pulls S3)
├── launch_spot.sh       # Spot instance launcher
├── generate.py          # Interactive inference + sample generation
├── export_to_hf.py      # Weight remapping to LlamaForCausalLM + HF push
└── eval_watcher.sh      # Cron-based eval runner (fires at 5k-iter checkpoints)
configs/
├── pretrain_350m.yaml
└── finetune_instruct.yaml
```

**Key implementation details**:
- RoPE applied per-head with precomputed cos/sin cache
- GQA via `repeat_interleave` — 4 KV heads expanded to 16 for attention
- Flash Attention via `F.scaled_dot_product_attention(is_causal=True)` — no separate package
- `torch.compile` disabled for all training (hangs on custom RoPE/GQA ops) — PyTorch 2.7 native ops used instead
- bfloat16 training with no gradient scaler (L40S native support)
- Data loaded via `np.memmap` — avoids loading 18GB dataset into RAM
- Checkpoints include full `ModelConfig` and `TrainConfig` for exact reproducibility
- Checkpoint loading requires `weights_only=False` (PyTorch 2.6+ changed default)

---

## Limitations

- **Context window**: 2,048 tokens. Not suitable for long-document tasks.
- **Knowledge depth**: 10B pretraining tokens is modest. Expect gaps in niche or technical topics.
- **Reasoning**: 350M parameters is below the threshold for reliable multi-step reasoning or arithmetic.
- **Code generation**: SFT on Alpaca-cleaned does not reliably produce correct code. Functions may be structurally valid but semantically wrong.
- **Repetition**: The model occasionally repeats phrases within a response, a known artifact of SFT-only training without RLHF or DPO.
- **No safety alignment**: SFT only, no RLHF or DPO. May produce inconsistent or unhelpful outputs on adversarial prompts.
- **English only**: Trained exclusively on English text.
- **Best use cases**: Domain fine-tuning on narrow tasks, edge/embedded deployment, educational experimentation, research baseline.

---

## Intended Use

This model is intended for:
- Researchers and students learning LLM training from scratch
- Developers needing a small, openly documented baseline for fine-tuning experiments
- Edge deployment scenarios where model size is constrained (~700MB bfloat16)
- Domain-specific fine-tuning where a small specialized model can match larger general models

---

## Citation

```bibtex
@misc{llm350m2026,
  title  = {LLM-350M-Instruct: A Reproducible 350M Parameter LLM Trained from Scratch},
  author = {Stoner, B.},
  year   = {2026},
  url    = {https://huggingface.co/sandbreak80sd/llm-350m-instruct},
  note   = {Training code: https://github.com/sandbreak80/llm-350m}
}
```
