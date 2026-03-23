#!/bin/bash
set -ex

echo "=== G6e (8x L40S) Benchmark Setup ==="
echo "Start time: $(date)"

# 1. System prep
sudo apt-get update -y && sudo apt-get install -y unzip wget git python3-venv python3-dev

# 2. Verify GPU
nvidia-smi
echo "=== GPU Topology ==="
nvidia-smi topo -m

# 3. NVMe already mounted by DLAMI at /opt/dlami/nvme
echo "NVMe storage:"
df -h /opt/dlami/nvme

# 4. Create Python venv (this DLAMI has Python 3.10, no conda)
python3 -m venv /home/ubuntu/pluto-env
source /home/ubuntu/pluto-env/bin/activate

# 5. Downgrade pip for omegaconf compatibility
pip install --upgrade pip
pip install "pip<24.1"

# 6. Install PyTorch 2.5.0 with CUDA 12.4 (compatible with CUDA 12.9 runtime)
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# 7. Verify PyTorch + CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_mem / 1024**3:.1f}GB, SM {props.major}.{props.minor}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
print(f'NCCL: {torch.cuda.nccl.version()}')
"

# 8. Install NATTEN (Neighborhood Attention for Pluto)
pip install natten==0.17.5+torch250cu124 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com || {
    echo "NATTEN prebuilt wheel failed, trying source build..."
    pip install natten==0.17.5 --no-build-isolation
}

# 9. Install nuplan-devkit
cd /home/ubuntu
if [ ! -d nuplan-devkit ]; then
    git clone https://github.com/motional/nuplan-devkit.git
fi
cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt
echo "=== NUPLAN_DEVKIT_DONE ==="

# 10. Install pluto
cd /home/ubuntu
if [ ! -d pluto ]; then
    git clone https://github.com/jchengai/pluto.git
fi
cd pluto

# Run pluto's setup but skip its PyTorch install (we already have 2.5)
# Just install pluto's other dependencies
pip install -e . || true
# Install remaining deps from pluto's setup_env.sh without torch reinstall
pip install pytorch-lightning==2.0.1 hydra-core==1.3.2 wandb l5kit shapely==2.0.1 || true
echo "=== PLUTO_DONE ==="

# 11. Apply Pluto code patches for BF16
cd /home/ubuntu/pluto
ENCODER_FILE="src/models/pluto/modules/agent_encoder.py"
if [ -f "$ENCODER_FILE" ]; then
    grep -q "dtype=x_agent_tmp.dtype" "$ENCODER_FILE" || \
    sed -i 's/torch\.zeros(num_agent_type, self\.d_model, device=x_agent_tmp\.device)/torch.zeros(num_agent_type, self.d_model, device=x_agent_tmp.device, dtype=x_agent_tmp.dtype)/g' "$ENCODER_FILE"
    echo "Patched $ENCODER_FILE for BF16 dtype compatibility"
fi

# Create missing __init__.py files
find src -type d -exec sh -c 'touch "$1/__init__.py" 2>/dev/null' _ {} \;

# 12. Copy code to NVMe (Ray workers need access)
cp -r /home/ubuntu/pluto /opt/dlami/nvme/pluto 2>/dev/null || true
cp -r /home/ubuntu/nuplan-devkit /opt/dlami/nvme/nuplan-devkit 2>/dev/null || true

# 13. Create data directories (symlink to NVMe)
sudo mkdir -p /opt/dlami/nvme/nuplan/dataset
sudo mkdir -p /opt/dlami/nvme/nuplan/exp
sudo chown -R ubuntu:ubuntu /opt/dlami/nvme/nuplan
sudo ln -sfn /opt/dlami/nvme/nuplan /nuplan

# 14. Final verification
echo "=== Final Environment Check ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPUs: {torch.cuda.device_count()}')
print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}')
print(f'NCCL version: {torch.cuda.nccl.version()}')
try:
    import natten
    print(f'NATTEN: {natten.__version__}')
except:
    print('NATTEN: NOT INSTALLED')
"

echo "=== ALL_DONE ==="
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Download data:  source ~/pluto-env/bin/activate && bash download_data.sh"
echo "  2. Run benchmark:  bash run_benchmark_g6e.sh"
