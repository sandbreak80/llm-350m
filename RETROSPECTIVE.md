# LLM-350M Project Retrospective

*Closing notes on the 350M parameter LLM project. Written after V2 published and instance terminated.*

---

## What We Built

A 350M parameter instruction-tuned language model, trained from scratch on a single rented GPU for under $500, published to HuggingFace in two iterations. The goal was to understand how LLMs actually work by building one end-to-end — not fine-tuning someone else's weights, but pretraining on raw text and learning every layer of the stack firsthand.

Two published models:
- **[llm-350m-instruct](https://huggingface.co/sandbreak80sd/llm-350m-instruct)** — V1, Alpaca SFT, val loss 1.7189
- **[llm-350m-instruct-v2](https://huggingface.co/sandbreak80sd/llm-350m-instruct-v2)** — V2, OpenHermes SFT, val loss 1.3704

---

## Results

| Benchmark | V1 (Alpaca) | V2 (OpenHermes) | Δ |
|---|---|---|---|
| HellaSwag | 38.40% | 37.60% | -0.80% |
| LAMBADA | 34.00% | 35.30% | +1.30% |
| ARC-Easy | 58.20% | 58.40% | +0.20% |
| ARC-Challenge | 27.76% | 25.42% | -2.34% |
| WinoGrande | 52.80% | 52.40% | -0.40% |
| **Val Loss** | **1.7189** | **1.3704** | **-20.3%** |

The benchmark deltas are small and mixed. Val loss is the more honest signal at this scale — a 20.3% improvement is substantial. The benchmarks are noisy at 350M parameters and sensitive to prompt format changes (V1 uses Alpaca, V2 uses ChatML — different token distributions affect loglikelihood scoring).

---

## Architecture

Modern LLaMA-style at 350M. Every component was a deliberate upgrade over the nanoGPT/GPT-2 baseline:

| Component | Choice | Why |
|---|---|---|
| Positional encoding | RoPE | Better length generalization |
| Normalization | RMSNorm, pre-norm | More stable than post-LayerNorm |
| Activation | SwiGLU | Better than GELU; intermediate_size = ~2.75x hidden (gating offsets the expansion) |
| Attention | GQA (4 KV / 16 Q heads) | 4x fewer KV parameters; LLaMA-3 style |
| Context | 2048 tokens | 2x GPT-2 |
| Attention impl | Flash Attention 2 (SDPA) | Memory efficient, no separate package |
| Embeddings | Tied (embed = lm_head) | Saves ~50M params, improves training signal |

Config: `n_layers=24, n_heads=16, n_kv_heads=4, n_embd=1024, intermediate_size=2816, vocab_size=50304`

---

## Training Summary

**Pretraining**: ~10B tokens from FineWeb-Edu, 60K iters, LR 6e-4, bfloat16, g6e.xlarge (L40S 46GB). ~$290.

**V1 SFT**: yahma/alpaca-cleaned (52K, GPT-3.5), Alpaca format, 1,500 iters, LR 2e-5, 2,500 FineWeb anti-forgetting. ~$8.

**V2 SFT**: teknium/OpenHermes-2.5 (200K, GPT-4), ChatML format, 4,000 iters, LR 1e-5, 10K FineWeb anti-forgetting. Best checkpoint iter 3800. ~$12.

**Total project cost: ~$310.**

---

## What Worked

**ChatML + GPT-4 data** was the single biggest lever. Going from 52K GPT-3.5 Alpaca examples to 200K GPT-4 ChatML examples dropped val loss 20.3%. Format matters too: ChatML is cleaner, more consistent, and better suited to modern inference tooling.

**Anti-forgetting blend** — mixing FineWeb samples during SFT prevented catastrophic forgetting. V2's larger blend (10K vs 2,500) contributed to stability at 4,000 iters.

**Loss masking on assistant turns only** — critical for SFT quality. Training on only the model's own outputs is what separates instruction tuning from continued pretraining.

**The architecture** — RoPE, RMSNorm, SwiGLU, GQA all worked cleanly. No instability, no surprises. LLaMA-style is robust even at 350M.

**Spot-resilient infrastructure** — SIGTERM handler, checkpoint every 500 iters, S3 sync via cron. The run survived multiple spot interruptions with minimal iteration loss.

**Non-blocking eval pipeline** — benchmarks ran on CPU via cron watcher while training occupied the GPU full-time. Zero GPU time wasted on evals.

**S3 completion marker pattern** — polling S3 for a marker file was a clean way to coordinate the local cleanup script with the remote pipeline.

---

## What Didn't Work (or Was Painful)

**Disk management** — The root EBS volume (30GB) fills fast between NVIDIA drivers (17GB) and PyTorch (8.6GB). Large files must go to the data volume from the start. Retrofitting symlinks mid-pipeline is messy. *Fix for next project: mount /data at launch and set all large output paths there explicitly.*

**Fine-grained HF tokens** — Both tokens on the instance were read-only fine-grained tokens. The push failed with 403 for hours before we realized. *Fix: always use a simple "write" role token; verify with `whoami()` before starting any pipeline.*

**`weights_only=False`** — PyTorch 2.6+ changed the default for `torch.load`. Checkpoint loading silently broke. *Fix: add to CLAUDE.md as a known gotcha — done.*

**Tied embeddings in safetensors** — `save_file` raises `RuntimeError` when two tensors share memory. Solution: omit `lm_head.weight` and set `tie_word_embeddings=True` — HF reconstructs it automatically. Non-obvious. *Fix: document in export script — done.*

**`torch.compile` disabled for pretraining** — Hangs on custom RoPE/GQA ops. Left off despite the potential speedup. Finetuning with compile worked. *Fix for next project: invest in making compile work with the RoPE kernel, or switch to a native RoPE implementation.*

**Bootstrap running pretrain data prep** — `aws_setup.sh` detected missing training data and started streaming 18GB. Needed `pkill` and a targeted re-run. *Fix: add `--skip-data` flag to bootstrap, or make data prep a separate explicit step.*

**llama.cpp GGUF conversion needs the right Python** — `convert_hf_to_gguf.py` requires `transformers`, which is only in `/opt/pytorch/bin/python`, not system `python3`. *Fix: always prefix with explicit Python path in pipeline scripts.*

---

## Cost

| Phase | Instance | Cost |
|---|---|---|
| Pretraining | g6e.xlarge (L40S 46GB) | ~$290 |
| V1 SFT | g6e.xlarge | ~$8 |
| V2 SFT + full pipeline | g6e.xlarge | ~$12 |
| **Total** | | **~$310** |

Well within the $1,000 budget. The project proved a capable instruction-following model can be built from scratch for well under $500 in pure training cost.

---

## What the Next Project Should Do Differently

1. **More pretraining data diversity** — FineWeb-Edu is high quality but narrow (educational/web). For a 1B+ model, DCLM-Baseline or a Dolma mix will produce a more capable base with broader world knowledge.

2. **Chinchilla-optimal or intentional over-training** — At 1B+, the token count decision matters more. Either train Chinchilla-optimal (~20B tokens for 1B params) or deliberately over-train for inference efficiency (LLaMA-style). Make this a conscious choice upfront.

3. **DPO from the start** — SFT alone leaves significant quality on the table. OpenHermes-2.5 has DPO-ready preference pairs. Plan for a DPO stage after SFT; don't treat it as an afterthought.

4. **Multi-GPU / gradient checkpointing** — 1B+ won't fit in 24GB without one of these. DDP or gradient checkpointing needs to be built into the training loop from day one, not bolted on later.

5. **`torch.compile` for pretraining** — Worth investing time to make it work. At 1B+, the 10-30% speedup meaningfully reduces training cost. Likely requires rewriting the RoPE kernel to use PyTorch native ops.

6. **Write token in SSM from day one** — Store a valid HF write-role token in SSM alongside the read token. Verify it before launch, not mid-pipeline.

7. **Explicit data volume paths in all scripts** — Never assume root volume has space. Every script that writes large files should use `/data/` explicitly with paths configured at the top of the file.

---

## Final State

- **GitHub**: `sandbreak80/llm-350m` — full training codebase, Apache 2.0
- **V1 HF**: `sandbreak80sd/llm-350m-instruct` — safetensors + GGUF Q4_K_M/Q8_0/F16
- **V2 HF**: `sandbreak80sd/llm-350m-instruct-v2` — safetensors + GGUF Q4_K_M/Q8_0/F16 + Modelfile
- **S3**: `finetune/best.pt` (V1) and `finetune_v2/best.pt` (V2) preserved; everything else cleaned up
- **Instance**: terminated
- **Total cost**: ~$310
