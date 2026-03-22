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
- 可通过增大 batch_size 或使用更大模型进一步压榨 GPU

## 文件清单

| 文件 | 用途 |
|------|------|
| setup.sh | 初始环境搭建脚本 |
| setup_fix.sh | 修复 pip/conda 兼容性问题 |
| download_data.sh | nuPlan 数据集下载脚本 |
| run_benchmark.sh | Benchmark 脚本 v1 |
| run_benchmark_v2.sh | Benchmark 脚本 v2（当前使用） |
| h200-benchmark.pem | SSH 密钥 |

## SSH 访问

```bash
ssh -i h200-benchmark.pem ubuntu@<PUBLIC_IP>

# 检查训练进度
tail -f /home/ubuntu/benchmark_v5.log

# 检查 GPU 状态
nvidia-smi

# 查看 checkpoint
ls -lh /nuplan/exp/exp/training/pluto/2026.03.22.10.27.43/checkpoints/

# 查看 GPU 监控日志
cat /home/ubuntu/gpu_monitor.log
```
