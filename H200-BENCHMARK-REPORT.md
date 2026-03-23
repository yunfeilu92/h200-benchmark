# H200 训练效率 Benchmark 报告

## 项目概述

验证 AWS H200 GPU 训练效率，使用自动驾驶规控算法 Pluto + nuPlan 数据集。

## 基础设施

### 实例信息

| 项目 | 详情 |
|------|------|
| 实例类型 | p5en.48xlarge (8x NVIDIA H200) |
| Region | us-east-2 (Ohio) |
| AZ | us-east-2a |
| Instance ID | i-01e4d3da3da15af1c |
| AMI | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5.1 (Ubuntu 22.04) |
| AMI ID | ami-0ead9b3b0f669a3e0 |

### Capacity Block

| 项目 | 详情 |
|------|------|
| Reservation ID | cr-0de24f5243e693bd6 |
| 开始时间 | 2026-03-22 07:46 UTC |
| 结束时间 | 2026-03-26 11:30 UTC |
| 时长 | ~100 小时 |
| 费用 | $4,150.10 |

### GPU 硬件

| 项目 | 详情 |
|------|------|
| GPU | 8x NVIDIA H200 |
| 显存 | 144 GB HBM3e / 每卡 |
| 驱动 | NVIDIA 580.95.05 |
| CUDA | 12.4 (系统) / 11.8 (PyTorch) |

### 存储

| 挂载点 | 类型 | 容量 | 用途 |
|--------|------|------|------|
| /opt/dlami/nvme | 本地 NVMe SSD (8x3.5TB LVM) | 27.6 TB | 数据集、代码、cache（所有 IO 密集操作） |
| / (EBS gp3) | EBS | 1 TB | 系统、conda 环境 |

> **注意**：所有训练相关的数据、代码、缓存均放在本地 NVMe 上，严禁使用 EBS，避免 IO 瓶颈影响 benchmark 结果。`/nuplan` 是指向 `/opt/dlami/nvme/nuplan` 的软链接。

## 算法 & 数据

### Pluto 算法

| 项目 | 详情 |
|------|------|
| 仓库 | https://github.com/jchengai/pluto |
| 论文 | PLUTO: Push the Limit of Imitation Learning-based Planning for Autonomous Driving (arXiv:2404.14327) |
| 任务 | 自动驾驶规控（Planning） |
| 模型结构 | Transformer encoder-decoder, dim=128, 4层 encoder + 4层 decoder, 12 modes |
| 框架 | PyTorch 2.0.1 + PyTorch Lightning 2.0.1 |
| 关键依赖 | nuplan-devkit, NATTEN 0.14.6 (Neighborhood Attention) |

### nuPlan 数据集

| 项目 | 详情 |
|------|------|
| 来源 | s3://motional-nuplan/public/nuplan-v1.1/ (ap-northeast-1) |
| 格式 | SQLite 数据库 (.db)，每个约 16MB |
| 数据内容 | ego_pose, lidar_box, track, scene, traffic_light_status, 高精地图 |

#### 数据库表结构（12 张表）

| 表 | 内容 | 关键字段 |
|---|---|---|
| ego_pose | 自车位姿 | x/y/z, 四元数旋转, 速度/加速度/角速度 |
| lidar_pc | 激光雷达点云帧 | 文件名指针, 时间戳 |
| lidar_box | 3D 检测框 | 位置/尺寸/速度/航向角/置信度 |
| track | 目标轨迹 | 类别, 宽/长/高 |
| image | 图像帧 | 8 个相机 |
| camera | 相机参数 | 内参/外参/畸变 |
| scene | 场景定义 | 目标位姿, 路段 ID |
| scenario_tag | 场景标签 | 跟车、变道等 |
| traffic_light_status | 红绿灯状态 | lane_connector_id, status |
| category | 目标类别 | 车/人/自行车等（7 类） |
| log | 日志元数据 | 车辆、日期、地图版本 |
| lidar | 激光雷达参数 | 通道、模型、外参 |

#### 下载的数据子集

| 子集 | 文件数 | 大小 | 状态 |
|------|--------|------|------|
| nuplan-maps-v1.0 | - | 1.4 GB | ✅ 已下载解压 |
| nuplan-v1.1_mini | - | 8 GB (14 GB 解压) | ✅ 已下载解压 |
| train_boston | 1,182 | 36 GB (44 GB 解压) | ✅ 已下载解压 |
| train_pittsburgh | 1,429 | 29 GB (45 GB 解压) | ✅ 已下载解压 |
| train_singapore | 1,563 | 33 GB (38 GB 解压) | ✅ 已下载解压 |
| train_vegas_1 | - | 144 GB | 🔄 下载中 |
| val | 376 | 90 GB (33 GB 解压) | ✅ 已下载解压 |
| **合计** | **4,550+** | **~360 GB** | |

> 地图覆盖 4 个城市：Boston, Pittsburgh, Las Vegas, Singapore

## 环境搭建

### 软件栈

```
conda env: pluto (Python 3.9)
├── PyTorch 2.0.1+cu118
├── torchvision 0.15.2
├── NATTEN 0.14.6+torch200cu118
├── PyTorch Lightning 2.0.1
├── nuplan-devkit (pip install -e)
├── timm, tensorboard, wandb, numba
└── LD_PRELOAD=/opt/conda/envs/pluto/lib/libstdc++.so.6 (兼容性修复)
```

### 遇到的问题 & 解决方案

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| omegaconf 安装失败 | pip >= 24.1 不接受非标准版本号 | `pip install "pip<24.1"` |
| `libstdc++.so.6: CXXABI_1.3.15 not found` | 系统 libstdc++ 版本低于 conda 内 ICU 库要求 | `export LD_PRELOAD=/opt/conda/envs/pluto/lib/libstdc++.so.6` |
| `src.custom_training` 导入失败 | 缺少 `__init__.py` 文件 | 递归创建所有 `__init__.py` |
| NATTEN 编译失败 | CUDA 版本不匹配 | 从 shi-labs.com 下载预编译 wheel（需 `--trusted-host` 绕过过期 SSL） |
| `database disk image is malformed` | 磁盘满时解压导致 3 个 db 文件损坏 | 删除 3 个损坏文件（3/4881） |
| maps 路径找不到 | NUPLAN_MAPS_ROOT 指向错误层级 | 修改为 `/nuplan/dataset/maps/nuplan-maps-v1.0` |
| Ray worker 找不到 Pluto 模块 | worker 进程不继承 PYTHONPATH | `export PYTHONPATH=/opt/dlami/nvme/pluto:$PYTHONPATH` |

## 训练流程

### 数据处理 Pipeline

```
原始 SQLite DB
    ↓ ScenarioBuilder (nuplan-devkit)
场景列表 (25,000 个场景)
    ↓ ScenarioFilter (training_scenarios_1M)
    ↓ PlutoFeatureBuilder (120m 范围, 48 agents, 8s 未来)
    ↓ FeaturePreprocessor → 序列化到 NVMe
Feature Cache (~400GB)
    ↓ DataLoader (batch_size=64, num_workers=16)
    ↓ DDP 8x H200
训练 (25 epochs)
```

### Feature Cache

| 项目 | 详情 |
|------|------|
| 目的 | 预计算训练特征，避免训练时重复 CPU 计算 |
| 场景数量 | 25,000 |
| Worker 数量 | 40 (Ray parallel) |
| 耗时 | ~2.5 小时 |
| Cache 大小 | ~400 GB |
| Cache 路径 | /nuplan/exp/cache_pluto (NVMe) |

#### Feature Builder 提取的特征

| 特征 | 参数 |
|------|------|
| 感知范围 | 120m |
| 历史 | 2 秒, 21 帧 |
| 未来 (GT) | 8 秒, 80 步 (sample_interval=0.1s) |
| Agent 输入 | 最多 48 个, 6 通道状态 |
| 静态障碍物 | 最多 10 个 |
| 地图多边形 | 6 通道 |
| 参考线 | 从高精地图构建 |

### 训练配置

| 参数 | 值 |
|------|-----|
| GPU | 8x H200 (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7) |
| 策略 | DDP (ddp_find_unused_parameters_false) |
| 精度 | FP32 |
| Batch Size | 64 (总), 8/卡 |
| Epochs | 25 |
| 学习率 | 1e-3 (Cosine + 3 epoch warmup) |
| 优化器 | Adam |
| Weight Decay | 0.0001 |
| Gradient Clip | 5.0 (norm) |
| DataLoader Workers | 16 |

### 训练指标（进行中）

| 时间点 | Epoch | 状态 |
|--------|-------|------|
| 10:30 UTC | 0 | 训练开始 |
| 11:09 UTC | 0 | Epoch 0 完成 (~39 min/epoch) |
| 11:48 UTC | 1 | Epoch 1 完成 |
| 预计完成 | 25 | ~2026-03-23 02:30 UTC |

### GPU 利用率快照（Epoch 2 进行中）

| GPU | 利用率 | 显存使用 | 温度 | 功耗 |
|-----|--------|----------|------|------|
| 0 | 57% | 22 GB / 144 GB | 36°C | 143W / 700W |
| 1 | 96% | 22 GB / 144 GB | 33°C | 143W / 700W |
| 2 | 82% | 24 GB / 144 GB | 36°C | 144W / 700W |
| 3 | 64% | 29 GB / 144 GB | 34°C | 147W / 700W |
| 4 | 62% | 24 GB / 144 GB | 35°C | 148W / 700W |
| 5 | 88% | 25 GB / 144 GB | 36°C | 142W / 700W |
| 6 | 60% | 18 GB / 144 GB | 38°C | 145W / 700W |
| 7 | 94% | 18 GB / 144 GB | 35°C | 141W / 700W |

**初步观察**：
- GPU 利用率波动 57-96%，平均约 75%
- 显存仅使用 12-20%（18-29 GB / 144 GB）
- 功耗仅约 20% TDP（145W / 700W）
- 模型规模较小，未充分利用 H200 的计算和显存能力

## 优化迭代

### 优化 1: 增大 batch_size（64 → 384）

将总 batch_size 从 64 提升到 384（每卡 48），显存利用从 15% 提升到 55-75%。

### 优化 2: BF16 混合精度

从 FP32 切换到 BF16-mixed，需要升级环境：
- Python 3.9 → 3.10（NATTEN 0.17.5 要求）
- PyTorch 2.0.1+cu118 → 2.5.0+cu124（BF16 upsample 支持）
- NATTEN 0.14.6 → 0.17.5
- 修复 3 处 `torch.zeros` dtype 不匹配 + 1 处 `emb.weight` dtype 转换

### 多卡配置对比

| 配置 | Epoch 时间 | GPU 功耗 | 显存占用 | SM 利用率 |
|------|-----------|---------|---------|----------|
| 8 卡 FP32 bs=64 | ~39 min | ~145W | 18-29 GB (15%) | 50-96% |
| 8 卡 FP32 bs=384 | ~30 min | ~130W | 80-128 GB (60%) | 100% |
| 8 卡 BF16 bs=384 | **~21 min** | ~130W | 55-76 GB (45%) | 100% |
| 2 卡 BF16 bs=384 | ~70 min (含 val) | **~220W** | 58-80 GB (48%) | 100% |
| 1 卡 BF16 bs=384 | ~56 min | ~200W | 76-90 GB (57%) | 100% |

**关键发现**：
- 8 卡 DDP 并行效率仅 ~34%（理论 8 倍加速，实际 2.7 倍）
- 2 卡功耗最高（220W），DDP 开销最小
- BF16 vs FP32 提速 46%（39min → 21min）

## 性能瓶颈深度分析

### PyTorch Profiler 实测数据

使用 PyTorch Profiler 对单卡 BF16 bs=384 的 forward pass 做了 kernel 级分析（10 次前向推理）：

| 操作 | CUDA 时间 | 占比 | 类型 |
|------|----------|------|------|
| **upsample_linear1d** | **5051ms** | **89.4%** | 内存带宽操作（FPN 上采样） |
| layer_norm | 162ms | 2.9% | Element-wise |
| copy_ (dtype 转换) | 131ms | 2.3% | 内存操作 |
| baddbmm (Attention QKV) | 92ms | 1.6% | **Tensor Core** |
| _to_copy | 78ms | 1.4% | 内存操作 |
| NATTEN QK | 61ms | 1.1% | Tensor Core (Sm80 kernel) |
| bmm (矩阵乘) | 56ms | 1.0% | **Tensor Core** |
| softmax | 48ms | 0.8% | Element-wise |
| NATTEN AV | 37ms | 0.7% | Tensor Core |
| index | 34ms | 0.6% | 内存操作 |
| **总计** | **5651ms** | | |

### 模型基本信息

```
参数量: 4,058,689 (4M params, 7.7 MB BF16)
架构: dim=128, 4 encoder + 4 decoder, 4 heads, 12 modes
单步 Forward: 570.6 ms (bs=384)
吞吐量: 673 samples/sec
```

### 算力利用率

| 指标 | 实测值 | H200 峰值 | 利用率 |
|------|--------|----------|--------|
| MatMul TFLOPS | 14.8 | 1979 (BF16) | **0.75%** |
| MatMul 时间占比 | 2.6% | - | - |
| GPU 功耗 | 130-220W | 700W | 19-31% |

### 瓶颈根因

**89.4% 的 GPU 时间花在 `upsample_linear1d`（FPN 特征金字塔上采样）**，该操作位于 `src/models/pluto/layers/embedding.py:78`（NATSequenceEncoder 的 FPN 模块）。

```python
# embedding.py:78 - 占 89.4% GPU 时间的操作
laterals[i - 1] = laterals[i - 1] + F.interpolate(
    laterals[i],
    scale_factor=(...),
    mode="linear",        # 线性插值，不走 Tensor Core
    align_corners=False,
)
```

该操作特点：
1. **不走 Tensor Core**：线性插值是 element-wise 内存操作，只用 CUDA Core
2. **调用量大**：每个 forward 对 384×49=18,816 个 agent 序列分别做 FPN 上采样
3. **计算密度低**：每个元素只做 1 次乘法 + 1 次加法，但需要读写显存

**真正用到 Tensor Core 的矩阵运算（baddbmm + bmm）只占 2.6% 时间**，且 GEMM 维度仅 128×128，远小于 H200 的 528 个 Tensor Core 能并行处理的规模。

### 结论

H200 在 Pluto 上的低利用率**不是**简单的"模型太小"——而是模型架构中 89% 的计算时间花在了一个内存带宽瓶颈操作上，Tensor Core（H200 的核心优势）几乎没有参与。即使优化掉 upsample，dim=128 的 GEMM 也无法填满 H200 的 Tensor Core 阵列。

## 优化 3: FPN upsample linear → nearest

### 改动

`src/models/pluto/layers/embedding.py:78`：

```python
# Before: mode="linear" (89.4% GPU time)
# After:  mode="nearest" (8.9% GPU time)
F.interpolate(laterals[i], scale_factor=(...), mode="nearest")
```

### Profile 对比（单卡 BF16 bs=384，10 次 forward）

| 指标 | 优化前 (linear) | 优化后 (nearest) | 变化 |
|------|---------------|-----------------|------|
| Forward 时间 | 570.6 ms | **71.4 ms** | **8x 加速** |
| 吞吐量 | 673 samples/s | **5,375 samples/s** | **8x 提升** |
| upsample 时间占比 | 89.4% | 8.9% | 不再是瓶颈 |
| MatMul 时间占比 | 2.6% | 6.7% | 占比上升 |

### 训练速度对比（8 卡 BF16 bs=384）

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| Epoch 时间 | ~21 min | **~8 min** | **2.6x 加速** |
| 25 epochs 总计 | ~8.75 hr | **~3.3 hr** | |

### GPU 利用率变化

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| GPU 功耗 | ~130W (19% TDP) | **~245W (35% TDP)** | **+88%** |
| 显存带宽利用率 (mem%) | 0% | **4-47%** | 从零到显著 |
| SM 利用率 | 100%（kernel 空转为主） | 44-100%（波动更大但更真实） | |
| 显存 | 55-76 GB | 64-79 GB | 略增 |

### 全配置最终对比

| 配置 | Epoch 时间 | 功耗 | 相对速度 |
|------|-----------|------|---------|
| 8 卡 FP32 bs=64 (原始) | ~39 min | ~145W | 1x |
| 8 卡 FP32 bs=384 | ~30 min | ~130W | 1.3x |
| 8 卡 BF16 bs=384 | ~21 min | ~130W | 1.9x |
| **8 卡 BF16 bs=384 + nearest** | **~8 min** | **~245W** | **4.9x** |

### 剩余瓶颈

优化 upsample 后功耗从 130W 升到 245W（35% TDP），但距离 700W 满载仍有差距。根因是 dim=128 的 GEMM 太小，Tensor Core 无法填满。这是模型架构限制，在不改模型的前提下无解。

## Nearest vs Linear 模型精度对比（进行中）

为验证 FPN upsample 从 linear 改为 nearest 对模型效果的影响，进行串行 A/B 对比实验。

### 实验设置

- 条件完全相同：8 卡 BF16 bs=384，10 epochs，warmup=3
- 串行执行（避免 IO 争抢导致不公平）：先 nearest → 再 linear
- Loss 从 checkpoint 的 ModelCheckpoint callback 中提取

### Nearest 结果（已完成，76 分钟）

| Epoch | val_loss |
|-------|---------|
| 5 | 5.184 |
| 6 | 4.891 |
| 7 | 4.819 |
| 8 | 4.870 |
| 9 | **4.818** |

训练时间：06:38 → 07:54 UTC = 76 分钟（10 epochs）

### Linear 结果

进行中，预计约 3.5 小时完成。

### DataLoader 配置检查

| 参数 | 当前值 | 最优值 | 说明 |
|------|--------|--------|------|
| pin_memory | **True** ✅ | True | 已通过 `${gpu}` 自动开启 |
| persistent_workers | 未设置 | True | 每 epoch 重建 worker 有开销 |
| prefetch_factor | 默认 2 | 4 | 可提前准备更多 batch |
| num_workers | 32 | 32 | 已设较高值 |

pin_memory 已生效，persistent_workers 和 prefetch_factor 可后续优化。

### 进一步优化方向

| 优化项 | 预期收益 | 工作量 |
|--------|---------|--------|
| persistent_workers + prefetch_factor | ~10-15% 训练加速 | 配置修改 |
| torch.compile | ~10-20% | 几行代码 |
| 换更大模型 (dim=512+) | **根本解决 GPU 利用率** | 换算法 |
| 多任务并行 (不同模型分卡跑) | **整机利用率提升** | 运维 |

## 文件清单

| 文件 | 用途 |
|------|------|
| setup.sh | 环境搭建脚本（合并版） |
| download_data.sh | nuPlan 数据集下载脚本 |
| run_benchmark_v2.sh | Benchmark 训练脚本 |
| profile_pluto.py | GPU 性能分析脚本 |

## SSH 访问

```bash
ssh -i h200-benchmark.pem ubuntu@<PUBLIC_IP>

# 检查训练进度
tail -f /home/ubuntu/benchmark_bf16_v3.log

# 检查 GPU 状态
nvidia-smi

# 查看 GPU 监控日志
cat /home/ubuntu/gpu_monitor_bf16.log
```
