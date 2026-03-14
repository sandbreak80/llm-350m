# CLAUDE.md — LLM Training Project Rules

## Project Goal
Build a ~350M parameter instruction-tuned LLM from scratch and publish to HuggingFace.
Reference model: [Apex-1-Instruct-350M](https://huggingface.co/LH-Tech-AI/Apex-1-Instruct-350M)
**Target: Match or exceed Apex-1 quality using modern architecture best practices.**

## Hard Constraints
- **AWS budget: $1,000 USD max** — track spend aggressively
- **1 GPU preferred** (target: `g5.xlarge` = A10G 24GB @ ~$1/hr)
- All training code must work on a single GPU (DDP optional for future scale)
- Model must be publishable to HuggingFace under Apache 2.0

## Architecture Decisions (Modern > Apex-1 Reference)
We use modern improvements over the nanoGPT-style GPT-2 architecture in the reference model:

| Component | Apex-1 (Reference) | Our Model |
|---|---|---|
| Positional Encoding | Learned | **RoPE** (Rotary Position Embedding) |
| Normalization | LayerNorm (post) | **RMSNorm (pre-norm)** |
| Activation | GELU | **SwiGLU** |
| Attention | MHA | **GQA** (Grouped Query Attention) |
| Context Length | 1024 | **2048** |
| Attention Implementation | Standard | **Flash Attention 2** |
| Tokenizer | GPT-2 (tiktoken) | **GPT-2 (tiktoken) — same, proven** |
| Vocabulary Size | 50,304 | **50,304** |

These changes make the architecture closer to LLaMA-3/Mistral style at the same parameter count.

## Model Configuration (350M target)
```python
n_layers = 24
n_heads = 16
n_kv_heads = 4          # GQA: 4 kv heads, 16 query heads
n_embd = 1024
intermediate_size = 2816  # SwiGLU: ~2.75x hidden (not 4x, offset by gating)
block_size = 2048
vocab_size = 50304
```

## Dataset Strategy
### Pretraining (~10B tokens)
- Primary: `HuggingFaceFW/fineweb-edu` (10BT sample) — high quality educational web text
- Optional supplement: `allenai/dolma` or `cerebras/SlimPajama-627B` (sample)

### Instruction Finetuning
- `yahma/alpaca-cleaned` (~52K instructions)
- Anti-forgetting mix: 2,500 FineWeb samples (4:1 ratio — proven from reference)
- Optional upgrade: `teknium/OpenHermes-2.5` (higher quality, same format)

## AWS Setup
### Recommended Instance
- **Primary**: `g5.xlarge` — A10G 24GB VRAM, ~$1.006/hr on-demand
- **Budget estimate**: ~200 hrs training + ~100 hrs iteration = ~$300–$450 total
- **Region**: `us-east-1` or `us-west-2` (cheapest)
- **Spot instances**: Use for pretraining (up to 70% cheaper, save checkpoints every 500 iters)
- **Storage**: 100GB EBS gp3 for datasets + checkpoints

### AMI
Use the AWS Deep Learning AMI (PyTorch) — comes with CUDA, cuDNN, PyTorch pre-installed.
- AMI ID search: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch`

### Key Commands
```bash
# Launch spot instance
aws ec2 request-spot-instances --instance-type g5.xlarge ...

# Sync checkpoints to S3 (run every N minutes as cron)
aws s3 sync ./checkpoints s3://your-bucket/checkpoints/

# Monitor GPU
nvidia-smi -l 5
```

## Training Pipeline
1. `scripts/aws_setup.sh` — bootstrap instance (install deps, mount EBS, pull repo)
2. `python src/data/prepare.py` — download + tokenize datasets
3. `python src/training/train.py --config configs/pretrain_350m.yaml` — pretrain
4. `python src/training/finetune.py --config configs/finetune_instruct.yaml` — SFT
5. `python scripts/export_to_hf.py` — convert to HuggingFace format + push

## Code Conventions
- Python 3.11+, PyTorch 2.2+, Flash Attention 2
- Use `bfloat16` for training (A10G supports it natively)
- Type hints on all public functions
- Config via YAML files + dataclasses, not argparse soup
- All training state in checkpoints: model, optimizer, scheduler, iteration count
- Log to W&B (free tier) — track loss, lr, grad norm, tokens/sec
- No notebooks in `src/` — scripts only; notebooks in `notebooks/` for EDA only

## Cost Tracking Rules
- Before launching any AWS instance, estimate the cost and confirm it fits in budget
- Save spot instance interruption checkpoints every **500 iterations** minimum
- Use S3 for checkpoint storage (not EBS snapshots — cheaper)
- Terminate instances when not actively training — don't leave them idle

## HuggingFace Publishing Checklist
- [ ] Model card with training details, eval metrics, example outputs
- [ ] `config.json` compatible with `transformers` AutoModel
- [ ] Tokenizer files (`tokenizer.json`, `tokenizer_config.json`)
- [ ] Weights in `safetensors` format
- [ ] License: Apache 2.0
- [ ] Example inference code in model card
- [ ] Benchmark results (at minimum: loss curves, sample outputs)

## Key Files
| File | Purpose |
|---|---|
| `src/model/model.py` | Core LLM architecture |
| `src/model/config.py` | ModelConfig dataclass |
| `src/data/prepare.py` | Dataset download + tokenization |
| `src/training/train.py` | Pretraining loop |
| `src/training/finetune.py` | Instruction finetuning loop |
| `configs/pretrain_350m.yaml` | Pretraining hyperparameters |
| `configs/finetune_instruct.yaml` | Finetuning hyperparameters |
| `scripts/aws_setup.sh` | AWS instance bootstrap |
| `scripts/export_to_hf.py` | Export to HuggingFace format |
