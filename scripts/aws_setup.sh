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

# Note: flash-attn package not needed — we use F.scaled_dot_product_attention
# (PyTorch 2.0+ uses Flash Attention 2 kernels natively via SDPA)

# ── Credentials from SSM ─────────────────────────────────────────────────────
REGION="us-east-1"
S3_BUCKET="bstoner-llm-checkpoints-536277006919"

WANDB_KEY=$(aws ssm get-parameter --name "llm-training-wandb-key" --with-decryption --query Parameter.Value --output text --region $REGION 2>/dev/null || echo "")
HF_TOKEN=$(aws ssm get-parameter --name "llm-training-hf-token" --with-decryption --query Parameter.Value --output text --region $REGION 2>/dev/null || echo "")

if [ -n "$WANDB_KEY" ]; then
    /opt/pytorch/bin/wandb login "$WANDB_KEY" --relogin
    echo "W&B authenticated."
fi
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    /opt/pytorch/bin/python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null && echo "HuggingFace authenticated." || echo "HuggingFace login skipped (not required for checkpoint resume)."
fi

# ── Project setup ────────────────────────────────────────────────────────────
# Repo already cloned by user-data — just install it
cd /home/ec2-user/llm-project

# Make project importable
"$PIP" install -e . --quiet

# ── Storage setup ────────────────────────────────────────────────────────────
mkdir -p data/pretrain data/finetune checkpoints/pretrain checkpoints/finetune

# Pull checkpoints from S3 (resumes from latest if available)
echo "Pulling checkpoints from S3..."
aws s3 sync s3://${S3_BUCKET}/checkpoints/ checkpoints/ --quiet
echo "Checkpoints pulled: $(ls checkpoints/pretrain/*.pt 2>/dev/null | wc -l) files"

# Pull training data from S3 (avoids re-tokenizing 10BT on every restart)
echo "Pulling training data from S3..."
aws s3 sync s3://${S3_BUCKET}/data/ data/ --quiet
if [ -f data/pretrain/train.bin ]; then
    echo "Training data ready: $(du -sh data/pretrain/train.bin | cut -f1) train.bin"
else
    echo "No data in S3 yet — running prepare.py (this takes ~2 hours)..."
    export HF_HOME=/home/ec2-user/hf_cache
    export HF_DATASETS_CACHE=/home/ec2-user/hf_cache/datasets
    "$PYTHON" src/data/prepare.py --dataset all
    # Upload data to S3 for future restarts
    aws s3 sync data/ s3://${S3_BUCKET}/data/ --quiet
    echo "Training data uploaded to S3 for future restarts."
fi

# Set up S3 checkpoint sync cron (every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * aws s3 sync $(pwd)/checkpoints s3://${S3_BUCKET}/checkpoints/ --quiet") | crontab -
echo "Checkpoint sync cron installed → s3://${S3_BUCKET}/checkpoints/"

# ── Auto-start training ───────────────────────────────────────────────────────
echo ""
echo "Starting training in tmux session 'train'..."
tmux new-session -d -s train
tmux send-keys -t train \
    "export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
     cd /home/ec2-user/llm-project && \
     $PYTHON -u src/training/train.py --config configs/pretrain_350m.yaml 2>&1 | tee /tmp/train.log" \
    Enter

echo ""
echo "=== Setup complete — training started ==="
echo "  Attach to training: tmux attach -t train"
echo "  Watch log:          tail -f /tmp/train.log"
echo "  Monitor GPU:        watch -n5 nvidia-smi"
