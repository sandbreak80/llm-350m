#!/bin/bash
# Launch a spot instance for training.
# Spot instances are ~60-70% cheaper than on-demand — use for pretraining.
# IMPORTANT: ensure checkpoints are saved to S3 so they survive interruptions.
#
# Usage:
#   1. source config.env          # set required variables
#   2. bash scripts/launch_spot.sh
#
# Prerequisites: AWS CLI configured, key pair and security group already created.
# Find your AMI ID at: EC2 > Launch Instance > search "Deep Learning OSS Nvidia Driver AMI GPU PyTorch"

set -euo pipefail

: "${AWS_REGION:?AWS_REGION is not set. Run: source config.env}"
: "${AMI_ID:?AMI_ID is not set. Find yours in EC2 > AMI Catalog}"
: "${KEY_NAME:?KEY_NAME is not set (EC2 key pair name)}"
: "${SECURITY_GROUP_ID:?SECURITY_GROUP_ID is not set}"
: "${SUBNET_ID:?SUBNET_ID is not set}"
: "${S3_BUCKET:?S3_BUCKET is not set}"
: "${GITHUB_REPO:?GITHUB_REPO is not set (e.g. your-username/llm-350m)}"
: "${INSTANCE_TYPE:=g5.xlarge}"    # A10G 24GB VRAM, ~$1/hr on-demand
: "${SPOT_PRICE:=0.60}"            # Max bid; g5.xlarge on-demand ~$1.006/hr
: "${IAM_INSTANCE_PROFILE:=LLMTrainingProfile}"

# User data script — runs as root on instance boot
USER_DATA=$(cat <<USERDATA
#!/bin/bash
set -e
cd /home/ec2-user
sudo -u ec2-user git clone https://github.com/${GITHUB_REPO}.git llm-project
cd llm-project
export S3_BUCKET="${S3_BUCKET}" AWS_REGION="${AWS_REGION}"
sudo -u ec2-user bash scripts/aws_setup.sh
USERDATA
)

# Encode user data
ENCODED_USER_DATA=$(echo "$USER_DATA" | base64 -w 0)

echo "Launching spot instance..."
echo "Instance type: $INSTANCE_TYPE"
echo "Max price: \$$SPOT_PRICE/hr"
echo "Region: $AWS_REGION"

AWS="aws"

"$AWS" ec2 request-spot-instances \
    --region "$AWS_REGION" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification "{
        \"ImageId\": \"$AMI_ID\",
        \"InstanceType\": \"$INSTANCE_TYPE\",
        \"KeyName\": \"$KEY_NAME\",
        \"SubnetId\": \"$SUBNET_ID\",
        \"SecurityGroupIds\": [\"$SECURITY_GROUP_ID\"],
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
echo "  aws ec2 describe-spot-instance-requests --region $AWS_REGION"
echo ""
echo "Once running, SSH in and start training in tmux:"
echo "  tmux new -s train"
echo "  cd llm-project && python src/training/train.py --config configs/pretrain_350m.yaml"
