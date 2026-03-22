#!/bin/bash
set -ex

# Activate conda env
source /opt/conda/etc/profile.d/conda.sh
conda activate pluto

# Set paths - all on NVMe
export NUPLAN_DATA_ROOT=/nuplan/dataset
export NUPLAN_MAPS_ROOT=/nuplan/dataset/maps
export NUPLAN_EXP_ROOT=/nuplan/exp

cd /opt/dlami/nvme/pluto

echo "========================================="
echo "Step 1: Feature Cache (using available cities)"
echo "Start: $(date)"
echo "========================================="

python run_training.py \
    py_func=cache +training=train_pluto \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.cleanup_cache=true \
    scenario_filter=training_scenarios_1M \
    worker.threads_per_node=40 \
    2>&1 | tee /home/ubuntu/cache.log

echo "Cache done: $(date)"

echo "========================================="
echo "Step 2: Training on 8x H200"
echo "Start: $(date)"
echo "========================================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_training.py \
    py_func=train +training=train_pluto \
    worker=single_machine_thread_pool worker.max_workers=32 \
    scenario_builder=nuplan \
    cache.cache_path=/nuplan/exp/cache_pluto \
    cache.use_cache_without_dataset=true \
    data_loader.params.batch_size=64 \
    data_loader.params.num_workers=16 \
    lr=1e-3 epochs=25 warmup_epochs=3 weight_decay=0.0001 \
    lightning.trainer.params.accelerator=gpu \
    lightning.trainer.params.devices=8 \
    lightning.trainer.params.strategy=ddp_find_unused_parameters_false \
    lightning.trainer.params.precision=32 \
    wandb.mode=disabled \
    wandb.project=h200-benchmark \
    wandb.name=pluto-8xH200 \
    2>&1 | tee /home/ubuntu/train.log

echo "Training done: $(date)"
echo "========================================="
