#!/bin/bash
# local_cleanup.sh — Polls S3 for pipeline completion, then terminates
# the instance and cleans up storage.
#
# Run this locally after launching a training job:
#   source config.env
#   INSTANCE_ID=i-xxxxxxxxxxxx bash scripts/local_cleanup.sh &
#
# It runs in the background and handles everything automatically once the
# pipeline writes the completion marker to S3.

set -euo pipefail

: "${INSTANCE_ID:?INSTANCE_ID is not set. Pass it as: INSTANCE_ID=i-xxx bash scripts/local_cleanup.sh}"
: "${S3_BUCKET:?S3_BUCKET is not set. Run: source config.env}"
: "${AWS_REGION:=us-east-1}"

GH_TOKEN_FILE="/tmp/gh_token.txt"
LOG="/tmp/local_cleanup.log"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo "  LOCAL CLEANUP MONITOR"
echo "  $(date)"
echo "  Instance: $INSTANCE_ID"
echo "  Polling S3 for pipeline completion..."
echo "============================================"

# Poll until completion marker appears
while true; do
    if aws s3 ls "s3://$S3_BUCKET/pipeline_complete.txt" 2>/dev/null | grep -q "pipeline_complete"; then
        echo "$(date) — Completion marker found! Starting cleanup."
        break
    fi
    echo "$(date) — Not done yet, checking again in 10 minutes..."
    sleep 600
done

# Download and print the pipeline log
echo ""
echo "=== Pipeline Log Summary ==="
aws s3 cp "s3://$S3_BUCKET/pipeline.log" - 2>/dev/null | tail -30

# Terminate the instance
echo ""
echo "=== Terminating instance $INSTANCE_ID ==="
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text
echo "Instance terminating."

# S3 cleanup — delete intermediate files, keep checkpoints
echo ""
echo "=== S3 Cleanup ==="
aws s3 rm "s3://$S3_BUCKET/pipeline_complete.txt"
echo "Remaining S3 contents:"
aws s3 ls "s3://$S3_BUCKET/" --recursive --human-readable | grep -v "pipeline"

# Optional: update GitHub README (requires GH_TOKEN_FILE)
echo ""
echo "=== GitHub README update ==="
if [ -f "$GH_TOKEN_FILE" ] && [ -n "${GITHUB_REPO:-}" ] && [ -n "${GITHUB_USERNAME:-}" ]; then
    TOKEN=$(cat "$GH_TOKEN_FILE" | tr -d '[:space:]')
    cd "$REPO_DIR"
    git add README.md 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "Update README"
        git push "https://${GITHUB_USERNAME}:${TOKEN}@github.com/${GITHUB_REPO}.git" master
        echo "GitHub README updated."
    else
        echo "No README changes to push."
    fi
else
    echo "Skipping GitHub update (set GITHUB_USERNAME, GITHUB_REPO, and $GH_TOKEN_FILE to enable)."
fi

echo ""
echo "============================================"
echo "  ALL DONE — $(date)"
echo "  Instance terminated: $INSTANCE_ID"
echo "  Storage cleaned up."
echo "============================================"
