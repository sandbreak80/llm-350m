#!/bin/bash
# AWS instance bootstrap script
# Run once after launching a fresh Deep Learning AMI instance
# Usage: bash scripts/aws_setup.sh

set -euo pipefail

echo "=== LLM Training Environment Setup ==="

# ── System packages (Amazon Linux 2023) ──────────────────────────────────────
sudo dnf install -y git tmux htop

# ── Python dependencies ──────────────────────────────────────────────────────
pip install --upgrade pip

# Core ML stack — PyTorch 2.7 is pre-installed on the DLAMI, upgrade if needed
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Flash Attention 2 (requires CUDA toolkit — pre-installed on DLAMI)
pip install flash-attn --no-build-isolation

# Training dependencies
pip install \
    transformers \
    datasets \
    tiktoken \
    wandb \
    pyyaml \
    numpy \
    tqdm \
    safetensors \
    huggingface_hub

# ── Project setup ────────────────────────────────────────────────────────────
S3_BUCKET="bstoner-llm-checkpoints-536277006919"

# Clone repo
git clone https://github.com/sandbreak80/llm-350m.git llm-project
cd llm-project

# Make project importable
pip install -e .

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
echo "  3. Prepare data: python src/data/prepare.py --dataset all"
echo "  4. Start pretraining in tmux: tmux new -s train"
echo "     python src/training/train.py --config configs/pretrain_350m.yaml"
echo ""
echo "  Monitor GPU: nvtop"
echo "  Check costs: aws ce get-cost-and-usage ..."
