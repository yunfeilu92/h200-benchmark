#!/bin/bash
set -e

echo "=== H200 Benchmark Setup ==="
echo "Start time: $(date)"

# 1. System prep
sudo apt-get update -y && sudo apt-get install -y unzip wget git

# 2. Verify GPU
nvidia-smi

# 3. Create conda env
eval "$(conda shell.bash hook)"
conda create -n pluto python=3.9 -y
conda activate pluto

# 4. Install nuplan-devkit
cd /home/ubuntu
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt

# 5. Install pluto
cd /home/ubuntu
git clone https://github.com/jchengai/pluto.git
cd pluto
bash ./script/setup_env.sh
# setup_env.sh installs: torch 2.0.1+cu118, torchvision, natten, requirements.txt

# 6. Create data directory
sudo mkdir -p /nuplan/dataset
sudo mkdir -p /nuplan/exp
sudo chown -R ubuntu:ubuntu /nuplan

echo "=== Environment setup complete ==="
echo "End time: $(date)"
