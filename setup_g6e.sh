#!/bin/bash
set -ex

echo "=== G6e (8x L40S) Benchmark Setup ==="
echo "Start time: $(date)"

# 1. System prep
sudo apt-get update -y && sudo apt-get install -y unzip wget git

# 2. Verify GPU - 确认是 L40S
nvidia-smi
echo "=== GPU Topology (确认 PCIe 拓扑，无 NVLink) ==="
nvidia-smi topo -m

# 3. Setup NVMe storage (g6e.48xlarge has local NVMe SSDs)
# 检查 NVMe 设备
echo "=== NVMe Devices ==="
lsblk | grep nvme

# 如果 DLAMI 没有自动挂载 NVMe，手动配置
if [ ! -d /opt/dlami/nvme ] || [ "$(df /opt/dlami/nvme 2>/dev/null | tail -1 | awk '{print $2}')" = "0" ]; then
    echo "Setting up NVMe storage..."
    # 查找所有 NVMe 实例存储设备（排除 root volume）
    NVME_DEVICES=$(lsblk -d -n -o NAME,TYPE | grep disk | grep nvme | awk '{print "/dev/"$1}' | grep -v "$(df / | tail -1 | awk '{print $1}' | sed 's/p[0-9]*$//')" || true)

    if [ -n "$NVME_DEVICES" ]; then
        DEVICE_COUNT=$(echo "$NVME_DEVICES" | wc -l)
        if [ "$DEVICE_COUNT" -gt 1 ]; then
            # 多个 NVMe，用 LVM 合并
            sudo pvcreate $NVME_DEVICES
            sudo vgcreate nvme_vg $NVME_DEVICES
            sudo lvcreate -l 100%FREE -n nvme_lv nvme_vg
            sudo mkfs.xfs /dev/nvme_vg/nvme_lv
            sudo mkdir -p /opt/dlami/nvme
            sudo mount /dev/nvme_vg/nvme_lv /opt/dlami/nvme
        else
            # 单个 NVMe
            sudo mkfs.xfs -f $NVME_DEVICES
            sudo mkdir -p /opt/dlami/nvme
            sudo mount $NVME_DEVICES /opt/dlami/nvme
        fi
        sudo chown -R ubuntu:ubuntu /opt/dlami/nvme
    fi
fi

echo "NVMe storage:"
df -h /opt/dlami/nvme 2>/dev/null || echo "WARNING: No NVMe storage mounted"

# 4. Source conda
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"

# 5. Create conda env - Python 3.10 (与 H200 配置保持一致, nuplan-devkit 需要 <=3.10)
conda remove -n pluto310 --all -y 2>/dev/null || true
conda create -n pluto310 python=3.10 -y
conda activate pluto310

# 6. Downgrade pip for omegaconf compatibility
pip install "pip<24.1"

# 7. Install PyTorch 2.5.0 with CUDA 12.4 (L40S = Ada Lovelace sm_89, fully supported)
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124

# 8. Verify PyTorch + CUDA
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

# 9. Install NATTEN (Neighborhood Attention for Pluto)
# NATTEN 0.17.5 supports sm_89 (Ada Lovelace)
pip install natten==0.17.5+torch250cu124 -f https://shi-labs.com/natten/wheels/ --trusted-host shi-labs.com

# 10. Install nuplan-devkit
cd /home/ubuntu
if [ ! -d nuplan-devkit ]; then
    git clone https://github.com/motional/nuplan-devkit.git
fi
cd nuplan-devkit
pip install -e .
pip install -r ./requirements.txt
echo "=== NUPLAN_DEVKIT_DONE ==="

# 11. Install pluto
cd /home/ubuntu
if [ ! -d pluto ]; then
    git clone https://github.com/jchengai/pluto.git
fi
cd pluto
# 不运行 pluto 自带的 setup_env.sh (会装旧版 PyTorch)
# 手动安装 pluto 的其他依赖
pip install -e .
echo "=== PLUTO_DONE ==="

# 12. Apply Pluto code patches (与 H200 相同的 fixes)
cd /home/ubuntu/pluto

# Fix: agent_encoder.py dtype mismatch for BF16
ENCODER_FILE="src/models/pluto/modules/agent_encoder.py"
if [ -f "$ENCODER_FILE" ]; then
    # 在 torch.zeros 处加 dtype 参数
    sed -i 's/torch\.zeros(num_agent_type, self\.d_model, device=x_agent_tmp\.device)/torch.zeros(num_agent_type, self.d_model, device=x_agent_tmp.device, dtype=x_agent_tmp.dtype)/g' "$ENCODER_FILE"
    echo "Patched $ENCODER_FILE for BF16 dtype compatibility"
fi

# Create missing __init__.py files
find src -type d -exec sh -c 'touch "$1/__init__.py" 2>/dev/null' _ {} \;

# 13. Create data directories (symlink to NVMe)
if [ -d /opt/dlami/nvme ]; then
    sudo mkdir -p /opt/dlami/nvme/nuplan/dataset
    sudo mkdir -p /opt/dlami/nvme/nuplan/exp
    sudo chown -R ubuntu:ubuntu /opt/dlami/nvme/nuplan
    sudo ln -sfn /opt/dlami/nvme/nuplan /nuplan
    # 也把 pluto 源码放到 NVMe（Ray workers 需要访问）
    cp -r /home/ubuntu/pluto /opt/dlami/nvme/pluto 2>/dev/null || true
    cp -r /home/ubuntu/nuplan-devkit /opt/dlami/nvme/nuplan-devkit 2>/dev/null || true
else
    sudo mkdir -p /nuplan/dataset
    sudo mkdir -p /nuplan/exp
    sudo chown -R ubuntu:ubuntu /nuplan
fi

# 14. Final verification
echo "=== Final Environment Check ==="
python -c "
import torch
import natten
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'NATTEN: {natten.__version__}')
print(f'GPUs: {torch.cuda.device_count()}')
print(f'GPU 0: {torch.cuda.get_device_name(0)}')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}')
print(f'NCCL version: {torch.cuda.nccl.version()}')
"

echo "=== ALL_DONE ==="
echo "End time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Download data:  bash download_data.sh"
echo "  2. Run benchmark:  bash run_benchmark_g6e.sh"
