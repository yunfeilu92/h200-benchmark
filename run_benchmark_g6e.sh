#!/bin/bash
set -ex

echo "=== G6e (8x L40S) Benchmark ==="
echo "Start time: $(date)"

# Activate venv
source /home/ubuntu/pluto-env/bin/activate

# Set paths - all on NVMe
export NUPLAN_DATA_ROOT=/nuplan/dataset
export NUPLAN_MAPS_ROOT=/nuplan/dataset/maps/nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=/nuplan/exp

# Ensure pluto src is importable by Ray workers
export PYTHONPATH=/opt/dlami/nvme/pluto:$PYTHONPATH

# ============================================================
# NCCL 优化 - L40S PCIe 拓扑专用配置
# g6e 无 NVLink，GPU 通过 PCIe Gen4 x16 互联
# ============================================================
export NCCL_ALGO=Ring                    # Ring 在 PCIe 拓扑下通常最优
export NCCL_PROTO=Simple                 # Simple 协议在 PCIe 下延迟更低
export NCCL_MIN_NCHANNELS=4              # 增加并行通信通道
export NCCL_MAX_NCHANNELS=12
export NCCL_BUFFSIZE=8388608             # 8MB buffer，改善 PCIe 吞吐
export NCCL_P2P_LEVEL=PHB                # PCIe Host Bridge 级别 P2P
export CUDA_DEVICE_MAX_CONNECTIONS=1     # 改善 compute/communication overlap
export NCCL_NET=Socket                   # g6e 无 EFA，禁用 OFI 插件

# Ray worker GPU 可见性 - 防止 NATTEN import 时找不到 GPU
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export HYDRA_FULL_ERROR=1

# ============================================================
# 最优训练配置 (基于 CUDA profiling 结果):
#
# - precision=32 (FP32): BF16 在此模型无速度优势且 loss 更差
# - batch_size=128 (16/卡): FP32 下 44GB 显存安全上限
# - mode="nearest": Pluto embedding.py 需先修改 F.interpolate
#   (原始 mode="linear" 占 63.6% GPU 时间，是压倒性瓶颈)
#
# 修改 Pluto 代码 (embedding.py:81):
#   mode="linear" → mode="nearest"
#   删除 align_corners=False 行
# ============================================================

cd /opt/dlami/nvme/pluto

# Print GPU topology for reference
echo "=== GPU Topology ==="
nvidia-smi topo -m

echo "========================================="
echo "Step 1: Feature Cache"
echo "Start: $(date)"
echo "========================================="

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=32 \
    2>&1 | tee /home/ubuntu/cache_g6e.log

echo "Cache done: $(date)"

# Verify cache has data
CACHE_COUNT=$(find /nuplan/exp/cache_pluto -name "*.gz" -o -name "*.pkl" 2>/dev/null | wc -l)
echo "Cache files: $CACHE_COUNT"
if [ "$CACHE_COUNT" -eq 0 ]; then
    echo "ERROR: Cache is empty! Aborting training."
    exit 1
fi

echo "========================================="
echo "Step 2: Training on 8x L40S (FP32 + nearest)"
echo "Start: $(date)"
echo "========================================="

# Monitor GPU utilization in background
nvidia-smi dmon -s umt -d 10 > /home/ubuntu/gpu_monitor_g6e.log 2>&1 &
GPU_MON_PID=$!

export NCCL_DEBUG=WARN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_training.py \
    py_func=train +training=train_pluto \
    worker=single_machine_thread_pool worker.max_workers=16 \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=128 \
    data_loader.params.num_workers=16 \
    lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.devices=8 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_false \
    lightning.trainer.params.precision=32 \
    wandb.mode=disabled \
    wandb.project=g6e-benchmark \
    wandb.name=pluto-8xL40S-fp32-nearest \
    2>&1 | tee /home/ubuntu/train_g6e.log

kill $GPU_MON_PID 2>/dev/null

echo "Training done: $(date)"
echo "========================================="
echo "=== GPU Monitor Summary ==="
tail -20 /home/ubuntu/gpu_monitor_g6e.log

echo ""
echo "=== Benchmark Complete ==="
echo "Logs:"
echo "  Cache:  /home/ubuntu/cache_g6e.log"
echo "  Train:  /home/ubuntu/train_g6e.log"
echo "  GPU:    /home/ubuntu/gpu_monitor_g6e.log"
