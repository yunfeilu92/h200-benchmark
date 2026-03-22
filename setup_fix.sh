#!/bin/bash
set -ex

# Source conda properly
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"

# Remove broken env if exists
conda remove -n pluto --all -y 2>/dev/null || true

# Create fresh env
conda create -n pluto python=3.9 -y
conda activate pluto

# Downgrade pip for omegaconf compatibility
pip install "pip<24.1"

# Install nuplan-devkit
cd /home/ubuntu/nuplan-devkit
pip install -e .
pip install -r ./requirements.txt
echo "=== NUPLAN_DEVKIT_DONE ==="

# Clone and install pluto
cd /home/ubuntu
if [ ! -d pluto ]; then
    git clone https://github.com/jchengai/pluto.git
fi
cd pluto
bash ./script/setup_env.sh
echo "=== PLUTO_DONE ==="

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo "=== ALL_DONE ==="
