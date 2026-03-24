# MapQR H200 Benchmark 报告

## 项目概述

在 AWS p5en.48xlarge (8x NVIDIA H200) 上 benchmark MapQR (ECCV 2024) HD 地图构建模型，使用 nuScenes 数据集。对比不同 batch size、precision、data loading 配置下的训练性能。

**MapQR Repo**: https://github.com/HXMap/MapQR

## 基础设施

| 项目 | 详情 |
|------|------|
| 实例类型 | p5en.48xlarge |
| GPU | 8x NVIDIA H200 (141 GB HBM3e each, 528 Tensor Cores) |
| vCPU | 192 |
| RAM | 2048 GiB |
| 存储 | 27.6 TB NVMe SSD (/opt/dlami/nvme) |
| 网络 | 3200 Gbps EFA |
| 驱动 | NVIDIA 580.95.05, CUDA 13.0 (driver) / 12.4 (nvcc) |

## 软件栈

| 组件 | 版本 |
|------|------|
| PyTorch | 2.0.1+cu118 |
| mmcv-full | 1.7.2 (从源码编译, sm_90) |
| mmdet | 2.28.2 |
| mmseg | 0.30.0 |
| mmdet3d | 0.17.2 (from MapQR submodule) |
| Python | 3.9 (conda) |
| CUDA (PyTorch) | 11.8 |
| NumPy | 1.23.5 |

> 注：MapQR 原始代码基于 PyTorch 1.9 / CUDA 11.1。迁移到 H200 (sm_90) 需要 13+ 个兼容性 patch（详见下方）。

## 数据集

| 数据集 | 大小 | 说明 |
|--------|------|------|
| nuScenes v1.0-trainval | ~350 GB | 850 scenes (700 train / 150 val) |
| nuScenes v1.0-mini | ~5 GB | 10 scenes (快速验证) |
| CAN bus expansion | 4.4 GB | 车辆动力学数据 |
| Map expansion v1.3 | 457 MB | 高精地图 |

数据来源：`s3://motional-nuscenes/public/v1.0/`（公开，无需认证）

## Batch Size Benchmark

### 测试条件
- 8x H200 DDP 训练
- FP16 混合精度 (loss_scale=512)
- workers_per_gpu=4
- ResNet-50 backbone, 24 epochs
- Config: `mapqr_nusc_r50_24ep.py`

### 结果

| Batch/GPU | Total Batch | Time/iter | Iters/epoch | Throughput | GPU 显存 | ETA (24ep) |
|-----------|-------------|-----------|-------------|------------|----------|------------|
| **4** | 32 | 1.38s | 880 | **23.2 s/s** | 16.4 GB (11%) | 8:14 |
| **8** | 64 | 2.52s | 440 | **25.4 s/s** | 30.2 GB (21%) | 7:26 |
| **16** | 128 | - | 220 | - | - | im2col_step 不兼容 |
| **32** | 256 | 11.08s | 110 | **23.1 s/s** | 113.3 GB (79%) | 7:58 |

### 分析

- **batch=8 吞吐最高**（25.4 samples/s），比默认 batch=4 快 9.5%
- batch=32 虽然显存利用到 79%，但单 iter 极慢（11s vs 1.38s），总吞吐反而不如 batch=4
- batch=16 因 mmcv deformable attention 的 `im2col_step` 约束无法运行（`batch*6cameras % 64 != 0`）
- **瓶颈不在 GPU 计算，而在数据加载**（batch 翻倍，吞吐仅增 9.5%）

## Precision Benchmark (FP16 vs BF16)

### 测试条件
- batch=4/GPU, 8x H200

| 配置 | Time/iter | Data Time | Throughput | GPU 显存 |
|------|-----------|-----------|-----------|----------|
| **FP16 + workers=4** (baseline) | 1.346s | 0.118s | 23.8 s/s | 16,397 MB |
| **BF16 + workers=8** | 1.391s | 0.103s | 23.0 s/s | 16,399 MB |

### 分析

- **BF16 比 FP16 慢 ~3%**，原因是 mmcv 的 deformable attention CUDA kernel 内部使用 FP32，BF16 autocast 增加了额外的类型转换开销
- **workers=8 的 data_time 改善**（0.118→0.103s, 减少 13%）被 BF16 计算开销抵消
- 结论：**对 MapQR + mmcv 1.x 技术栈，FP16 是更好的选择**

## 8-GPU Profile（SM Utilization 分析）

### nvidia-smi dmon 采样（120 秒，1Hz）

```
Per-GPU SM Utilization:
 GPU |  SM Avg |  SM P50 |  SM P95 |  SM Max |   Mem% |  Power
-----+---------+---------+---------+---------+--------+-------
   0 |   45.9% |     42% |    100% |    100% |   6.7% |   195W
   1 |   49.8% |     46% |    100% |    100% |   6.3% |   191W
   2 |   42.2% |     38% |    100% |    100% |   6.2% |   195W
   3 |   48.8% |     47% |    100% |    100% |   6.0% |   191W
   4 |   50.4% |     52% |    100% |    100% |   6.9% |   197W
   5 |   49.6% |     49% |    100% |    100% |   6.3% |   194W
   6 |   45.7% |     45% |    100% |    100% |   6.1% |   194W
   7 |   46.6% |     46% |    100% |    100% |   4.8% |   190W

Overall SM: avg=47.4%, P50=45%, P95=100%, max=100%
```

### SM Utilization 分布

```
Idle   (<5%):   25.4%  ############
Low  (5-30%):   11.6%  #####
Mid (30-70%):   28.2%  ##############
High(70-95%):   14.3%  #######
Full (>=95%):   20.5%  ##########
```

### 时间分解

| 阶段 | 耗时 | 占比 |
|------|------|------|
| 数据加载 (CPU) | 0.243s | **25%** |
| GPU 计算 | 0.729s | 75% |
| 总 iter | 0.972s | 100% |

### 瓶颈分析

1. **25% 时间在等数据加载** — GPU idle 的主要原因
2. **Memory bandwidth 仅 5-7%** — 模型太小，远未用到 H200 的 4.8 TB/s HBM3e 带宽
3. **功耗 190-197W / 700W (28%)** — GPU 大量时间空转等待
4. **SM 利用率两极化** — 计算时 100%，但 25% 时间完全 idle（等数据/DDP 同步）
5. **H200 的 141GB 显存仅用了 11-21%** — MapQR (ResNet-50, dim=128) 对 H200 来说太小

## 优化配置

基于 profiling 结果，最优训练配置：

| 参数 | 默认值 | 优化值 | 原因 |
|------|--------|--------|------|
| samples_per_gpu | 4 | **8** | 吞吐提升 9.5% |
| workers_per_gpu | 4 | **8** | 减少数据等待 13% |
| persistent_workers | False | **True** | 避免每 epoch 重启 workers |
| prefetch_factor | 2 | **4** | 预加载更多 batch |
| CUDA_DEVICE_MAX_CONNECTIONS | - | **1** | 改善 DDP 通信重叠 |
| im2col_step | 64 | **48** | 兼容 batch=8 (8×6=48) |
| precision | FP16 | **FP16** | BF16 对 mmcv CUDA ops 无收益 |
| loss_scale | 512 (static) | 512 | 稳定 |

预估 ETA：~6.5-7h（vs 默认 ~8.5h，提升 ~20%）

## 迁移 Patch 清单（PyTorch 1.9 → 2.0 on H200）

| # | 问题 | 修复 |
|---|------|------|
| 1 | THC/THC.h 在 PyTorch 2.0 移除 | 替换为 `ATen/cuda/CUDAContext.h` + `c10/cuda/CUDAStream.h` |
| 2 | `at::cuda::getCurrentCUDAStream` → `c10::cuda` | 命名空间迁移 |
| 3 | `.type().is_cuda()` deprecated | 改为 `.is_cuda()` |
| 4 | mmcv CUDA version check (12.4 vs 11.8) | raise → warnings.warn |
| 5 | mmdet3d mmcv 版本限制 (<=1.4.0) | 放宽到 1.8.0 |
| 6 | SparseConv registry 冲突 (mmcv 1.7 vs mmdet3d) | `force=True` |
| 7 | EfficientNet + 78 个 plugin 注册冲突 | 全部 `force=True` |
| 8 | `numba.errors` → `numba.core.errors` | import 路径迁移 |
| 9 | `torch.distributed.launch` → `torchrun` | `--local-rank` 兼容 |
| 10 | multiprocessing spawn → fork | 解决 dict_keys pickle 问题 |
| 11 | libstdc++ CXXABI_1.3.15 缺失 | LD_LIBRARY_PATH 指向 conda lib |
| 12 | `absl-py` 2.4 用了 Python 3.10+ 语法 | 降级到 <2 |
| 13 | tensorboard `type \| None` 语法 | 降级到 <2.16 |
| 14 | shapely 2.x `intersects` API 变更 | 降级到 <2 |
| 15 | mmcv-full sm_90 kernel 缺失 | 从源码编译 `TORCH_CUDA_ARCH_LIST=9.0` |
| 16 | GeometricKernelAttention sm_90 | 重新编译 |
| 17 | nuScenes converter version 拼接错误 | patch `custom_nusc_map_converter.py` |

## 结论

1. **MapQR 在 H200 上严重 underutilized** — SM 平均利用率 47%，显存仅用 11-21%，功耗 28%
2. **瓶颈是数据加载而非 GPU 计算** — 25% 时间 GPU 完全空闲等数据
3. **batch=8 是最优 batch size** — 更大的 batch 因 attention 计算复杂度超线性增长反而更慢
4. **FP16 优于 BF16** — mmcv 的 deformable attention CUDA kernel 对 BF16 优化不足
5. **H200 更适合更大的模型**（更大的 backbone、更高分辨率、更多 BEV queries），MapQR ResNet-50 对它来说太轻量
