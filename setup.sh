#!/bin/bash
set -ex

echo "=== H200 Benchmark Setup ==="
echo "Start time: $(date)"

# 1. System prep
sudo apt-get update -y && sudo apt-get install -y unzip wget git

# 2. Verify GPU
nvidia-smi

# 3. Source conda properly (non-interactive SSH compatible)
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"

# 4. Create conda env (remove broken env if exists)
conda remove -n pluto --all -y 2>/dev/null || true
conda create -n pluto python=3.9 -y
conda activate pluto

# 5. Downgrade pip for omegaconf compatibility
pip install "pip<24.1"

# 6. Install nuplan-devkit
cd /home/ubuntu
if [ ! -d nuplan-devkit ]; then
    git clone https://github.com/motional/nuplan-devkit.git
fi
cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt
echo "=== NUPLAN_DEVKIT_DONE ==="

# 7. Install pluto
cd /home/ubuntu
if [ ! -d pluto ]; then
    git clone https://github.com/jchengai/pluto.git
fi
cd pluto
bash ./script/setup_env.sh
echo "=== PLUTO_DONE ==="

# 8. Create data directories
sudo mkdir -p /nuplan/dataset
sudo mkdir -p /nuplan/exp
sudo chown -R ubuntu:ubuntu /nuplan

# 9. Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
echo "=== ALL_DONE ==="
echo "End time: $(date)"
