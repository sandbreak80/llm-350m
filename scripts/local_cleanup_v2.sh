#!/bin/bash
# local_cleanup_v2.sh — Polls S3 for pipeline completion, then terminates
# instance and cleans up storage.
#
# Run this locally: bash scripts/local_cleanup_v2.sh &
# It will run in the background and handle everything automatically.

set -euo pipefail

INSTANCE_ID="i-01c55869150176725"
REGION="us-east-1"
S3_BUCKET="bstoner-llm-checkpoints-536277006919"
GH_TOKEN_FILE="/tmp/gh_token.txt"
LOG="/tmp/local_cleanup_v2.log"

exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  LOCAL CLEANUP MONITOR"
echo "  $(date)"
echo "  Polling S3 for pipeline completion..."
echo "============================================"

# Poll until completion marker appears
while true; do
    if aws s3 ls "s3://$S3_BUCKET/v2_pipeline_complete.txt" 2>/dev/null | grep -q "v2_pipeline_complete"; then
        echo "$(date) — Completion marker found! Starting cleanup."
        break
    fi
    echo "$(date) — Not done yet, checking again in 10 minutes..."
    sleep 600
done

# Download and print the pipeline log
echo ""
echo "=== Pipeline Log Summary ==="
aws s3 cp "s3://$S3_BUCKET/v2_pipeline.log" - 2>/dev/null | tail -30

# Terminate the instance
echo ""
echo "=== Terminating instance $INSTANCE_ID ==="
aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text
echo "Instance terminating."

# S3 cleanup — delete intermediate/large files, keep both checkpoints
echo ""
echo "=== S3 Cleanup ==="
aws s3 rm "s3://$S3_BUCKET/v2_pipeline_complete.txt"
echo "Kept:"
aws s3 ls "s3://$S3_BUCKET/" --recursive --human-readable | grep -v "pipeline"

# Update GitHub README to reference both V1 and V2
echo ""
echo "=== Updating GitHub README ==="
if [ -f "$GH_TOKEN_FILE" ]; then
    TOKEN=$(cat "$GH_TOKEN_FILE" | tr -d '[:space:]')
    REPO_DIR="/home/bstoner/code_projects/claude_setup/claude_setup"
    cd "$REPO_DIR"

    # Append V2 section to README if not already there
    if ! grep -q "llm-350m-instruct-v2" README.md 2>/dev/null; then
        cat >> README.md << 'EOF'

---

## V2 Model

V2 is published separately at [`sandbreak80sd/llm-350m-instruct-v2`](https://huggingface.co/sandbreak80sd/llm-350m-instruct-v2).

Key changes: OpenHermes-2.5 (200K GPT-4) instead of Alpaca-cleaned (52K GPT-3.5), ChatML format, lower LR (1e-5), 4000 training iterations.
EOF
        git add README.md
        git commit -m "Add V2 model reference to README"
        git push "https://sandbreak80:${TOKEN}@github.com/sandbreak80/llm-350m.git" master
        echo "GitHub README updated."
    else
        echo "GitHub README already has V2 reference."
    fi
else
    echo "No GitHub token found at $GH_TOKEN_FILE — skipping GitHub update."
fi

echo ""
echo "============================================"
echo "  ALL DONE — $(date)"
echo "  V2 published: https://huggingface.co/sandbreak80sd/llm-350m-instruct-v2"
echo "  Instance terminated."
echo "  Storage cleaned up."
echo "============================================"
