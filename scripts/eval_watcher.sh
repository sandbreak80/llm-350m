#!/bin/bash
# eval_watcher.sh — Watches for new checkpoints at 5k-iter multiples and runs benchmarks.
#
# Install as cron (runs every 5 minutes):
#   (crontab -l; echo "*/5 * * * * bash /home/ec2-user/llm-project/scripts/eval_watcher.sh") | crontab -
#
# Logs to: /tmp/eval.log
# Tracks evaluated checkpoints in: /tmp/eval_done.txt

set -euo pipefail

PROJ_DIR="/home/ec2-user/llm-project"
PYTHON="/opt/pytorch/bin/python"
CKPT_DIR="$PROJ_DIR/checkpoints/pretrain"
DONE_FILE="/tmp/eval_done.txt"
LOCK_FILE="/tmp/eval_watcher.lock"
LOG_FILE="/tmp/eval.log"

# Only one eval at a time
if [ -f "$LOCK_FILE" ]; then
    exit 0
fi
touch "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

touch "$DONE_FILE"
cd "$PROJ_DIR"

# Find checkpoints at 5k-iter multiples not yet evaluated, in ascending order
for ckpt in $(ls "$CKPT_DIR"/ckpt_*.pt 2>/dev/null | sort); do
    basename=$(basename "$ckpt" .pt)
    iter_str="${basename#ckpt_}"
    iter=$((10#$iter_str))

    # Only eval at 5k multiples (skip iter 0)
    if [ $((iter % 5000)) -ne 0 ] || [ "$iter" -eq 0 ]; then
        continue
    fi

    # Skip if already evaluated
    if grep -qxF "$ckpt" "$DONE_FILE" 2>/dev/null; then
        continue
    fi

    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') | Starting eval: iter $iter" | tee -a "$LOG_FILE"

    $PYTHON "$PROJ_DIR/src/eval/run_eval.py" \
        --checkpoint "$ckpt" \
        2>&1 | tee -a "$LOG_FILE"

    echo "$ckpt" >> "$DONE_FILE"
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') | Eval complete: iter $iter" | tee -a "$LOG_FILE"

    # One checkpoint per cron invocation — don't hog the slot
    break
done
