#!/bin/bash
set -e

echo "=== Downloading nuPlan dataset ==="
echo "Start time: $(date)"

DATA_DIR="/nuplan/dataset"
S3_BASE="s3://motional-nuplan/public/nuplan-v1.1"

cd $DATA_DIR

# Maps (required)
echo "Downloading maps..."
aws s3 cp ${S3_BASE}/nuplan-maps-v1.1.zip . --no-sign-request --region ap-northeast-1
unzip -q nuplan-maps-v1.1.zip -d maps && rm nuplan-maps-v1.1.zip

# Mini dataset (for sanity check)
echo "Downloading mini dataset..."
aws s3 cp ${S3_BASE}/nuplan-v1.1_mini.zip . --no-sign-request --region ap-northeast-1
unzip -q nuplan-v1.1_mini.zip -d mini && rm nuplan-v1.1_mini.zip

# Training data (selected cities)
echo "Downloading train boston..."
aws s3 cp ${S3_BASE}/nuplan-v1.1_train_boston.zip . --no-sign-request --region ap-northeast-1 &

echo "Downloading train pittsburgh..."
aws s3 cp ${S3_BASE}/nuplan-v1.1_train_pittsburgh.zip . --no-sign-request --region ap-northeast-1 &

echo "Downloading train singapore..."
aws s3 cp ${S3_BASE}/nuplan-v1.1_train_singapore.zip . --no-sign-request --region ap-northeast-1 &

echo "Downloading train vegas_1..."
aws s3 cp ${S3_BASE}/nuplan-v1.1_train_vegas_1.zip . --no-sign-request --region ap-northeast-1 &

# Validation data
echo "Downloading validation set..."
aws s3 cp ${S3_BASE}/nuplan-v1.1_val.zip . --no-sign-request --region ap-northeast-1 &

# Wait for all parallel downloads
wait

echo "Unzipping training data..."
for f in nuplan-v1.1_train_*.zip nuplan-v1.1_val.zip; do
  echo "Unzipping $f..."
  unzip -q "$f" && rm "$f" &
done
wait

echo "=== Download complete ==="
echo "End time: $(date)"
ls -lh $DATA_DIR
