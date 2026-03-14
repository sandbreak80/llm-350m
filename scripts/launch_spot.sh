#!/bin/bash
# Launch a spot instance for training.
# Spot instances are ~60-70% cheaper than on-demand — use for pretraining.
# IMPORTANT: ensure checkpoints are saved to S3 so they survive interruptions.
#
# Usage: bash scripts/launch_spot.sh
# Prerequisites: AWS CLI configured, key pair and security group created

set -euo pipefail

# ── Configuration — edit these ───────────────────────────────────────────────
REGION="us-east-1"
INSTANCE_TYPE="g5.xlarge"          # A10G 24GB VRAM, ~$1/hr on-demand
SPOT_PRICE="0.60"                  # Max bid (on-demand ~$1.006, spot ~$0.35-0.60)
AMI_ID="ami-0de4ae9106f688338"     # Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (AL2023) 20260307
KEY_NAME="llm-training-key"        # EC2 key pair (~/.ssh/llm-training-key.pem)
SECURITY_GROUP="sg-0debdff8b288db1fa"   # llm-training-sg (SSH from your IP only)
SUBNET_ID="subnet-0277da3975efdf31d"    # us-east-1f — g5 available here
S3_BUCKET="bstoner-llm-checkpoints-536277006919"
IAM_INSTANCE_PROFILE="LLMTrainingProfile"

# User data script — runs as root on instance boot
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -e
cd /home/ec2-user
sudo -u ec2-user git clone https://github.com/sandbreak80/llm-350m.git llm-project
cd llm-project
sudo -u ec2-user bash scripts/aws_setup.sh
USERDATA
)

# Encode user data
ENCODED_USER_DATA=$(echo "$USER_DATA" | base64 -w 0)

echo "Launching spot instance..."
echo "Instance type: $INSTANCE_TYPE"
echo "Max price: \$$SPOT_PRICE/hr"
echo "Region: $REGION"

AWS="/c/Program Files/Amazon/AWSCLIV2/aws.exe"

"$AWS" ec2 request-spot-instances \
    --region "$REGION" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SubnetId\": \"$SUBNET_ID\",
        \"SecurityGroupIds\": [\"$SECURITY_GROUP\"],
        \"IamInstanceProfile\": {\"Name\": \"$IAM_INSTANCE_PROFILE\"},
        \"UserData\": \"$ENCODED_USER_DATA\",
        \"BlockDeviceMappings\": [{
            \"DeviceName\": \"/dev/sda1\",
            \"Ebs\": {
                \"VolumeSize\": 150,
                \"VolumeType\": \"gp3\",
                \"Iops\": 3000,
                \"DeleteOnTermination\": true
            }
        }]
    }" \
    --spot-price "$SPOT_PRICE"

echo ""
echo "Spot request submitted. Check status:"
echo "  \"$AWS\" ec2 describe-spot-instance-requests --region $REGION"
echo ""
echo "Once running, SSH in and start training in tmux:"
echo "  tmux new -s train"
echo "  cd llm-project && python src/training/train.py --config configs/pretrain_350m.yaml"
