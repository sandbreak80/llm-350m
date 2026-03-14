#!/bin/bash
# AWS instance bootstrap script
# Run once after launching a fresh Deep Learning AMI instance
# Usage: bash scripts/aws_setup.sh

set -euo pipefail

echo "=== LLM Training Environment Setup ==="

# ── System packages (Amazon Linux 2023) ──────────────────────────────────────
sudo dnf install -y git tmux htop

# ── Python environment — DLAMI uses /opt/pytorch (Python 3.12, PyTorch 2.7) ──
PIP="/opt/pytorch/bin/pip"
PYTHON="/opt/pytorch/bin/python"

# PyTorch 2.7 + CUDA 12.8 are pre-installed — skip reinstall, just add deps
"$PIP" install --quiet \
    transformers \
    datasets \
    tiktoken \
    wandb \
    pyyaml \
    numpy \
    tqdm \
    safetensors \
    huggingface_hub

# Flash Attention 2
"$PIP" install flash-attn --no-build-isolation --quiet

# ── Project setup ────────────────────────────────────────────────────────────
S3_BUCKET="bstoner-llm-checkpoints-536277006919"

# Repo already cloned by user-data — just install it
cd /home/ec2-user/llm-project

# Make project importable
"$PIP" install -e . --quiet

# ── Storage setup ────────────────────────────────────────────────────────────
# Create data and checkpoint directories
mkdir -p data/pretrain data/finetune checkpoints/pretrain checkpoints/finetune

# Set up S3 checkpoint sync cron (every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * aws s3 sync $(pwd)/checkpoints s3://${S3_BUCKET}/checkpoints/ --quiet") | crontab -
echo "Checkpoint sync cron installed → s3://${S3_BUCKET}/checkpoints/"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Configure W&B: wandb login"
echo "  2. Configure HuggingFace: huggingface-cli login"
echo "  3. Prepare data: /opt/pytorch/bin/python src/data/prepare.py --dataset all"
echo "  4. Start pretraining in tmux: tmux new -s train"
echo "     /opt/pytorch/bin/python src/training/train.py --config configs/pretrain_350m.yaml"
echo ""
echo "  Monitor GPU: nvtop"
echo "  Check costs: aws ce get-cost-and-usage ..."
