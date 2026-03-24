# G6e (8x L40S) Benchmark Report

## 实例配置

| 项目 | 值 |
|------|-----|
| 实例类型 | g6e.48xlarge |
| GPU | 8x NVIDIA L40S, 45GB GDDR6 / 卡 |
| GPU 架构 | Ada Lovelace (SM 8.9) |
| 驱动 | NVIDIA 580.126.09 |
| CUDA | 12.4 (PyTorch) / 12.9 (系统) |
| PyTorch | 2.5.0+cu124 |
| NCCL | 2.21.5 |
| NATTEN | 0.17.5 |
| vCPU | 192 (AMD EPYC 7R13) |
| 内存 | 1536 GiB |
| 存储 | 6.9 TB NVMe SSD (LVM) |
| 网络 | 400 Gbps ENA (无 EFA) |
| GPU 互联 | PCIe Gen4 x16 (无 NVLink) |
| Region | us-east-1 |
| On-Demand 价格 | $30.13/hr |

## 软件栈

| 组件 | 版本 | 说明 |
|------|------|------|
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260317 |  |
| Python | 3.10.12 (系统 venv) | DLAMI 无 conda |
| PyTorch Lightning | 2.0.1 |  |
| torchmetrics | 0.11.4 | pluto 需要 compute_on_step |
| numpy | 1.23.5 | nuplan 需要 np.bool |
| NCCL_NET | Socket | g6e 无 EFA，禁用 OFI 插件 |
| NCCL_ALGO | Ring | PCIe 拓扑下最优 |

## Benchmark 任务

**算法**: [Pluto](https://github.com/jchengai/pluto) — 自动驾驶规控 (Planning) 模型，Transformer encoder-decoder，~4.1M 参数

**数据集**: nuPlan v1.1 — 577GB，8734 个 SQLite 数据库，4 个城市

**流程**:
1. Feature Cache: 32 个 Ray workers，31250 个 scenario → 485GB cache
2. DDP 训练: 8x L40S，3 epochs

## 训练结果

### BF16-mixed (bs=192, 24/卡)

| Epoch | 完成时间 (UTC) | 耗时 |
|-------|---------------|------|
| 0 | 10:55 | ~26 min |
| 1 | 11:20 | ~25 min |
| 2 | 11:45 | ~25 min |
| **总计** | | **~76 min** |

- 显存: 38-45 GB/卡
- 功耗: 100-140 W/卡 (28-40% TDP)
- 需要 4 处 dtype patch (torch.zeros 加 dtype 参数) 适配 BF16

### FP32 (bs=128, 16/卡)

| Epoch | 完成时间 (UTC) | 耗时 |
|-------|---------------|------|
| 0 | 16:25 | ~28 min |
| 1 | 16:51 | ~26 min |
| 2 | 17:16 | ~25 min |
| **总计** | | **~79 min** |

- 显存: 31-45 GB/卡
- 功耗: 167-195 W/卡 (48-56% TDP)
- 代码零改动

### 对比总结

| 配置 | 每卡 batch | Epoch 时间 | 3 Epoch 总计 | 代码改动 |
|------|-----------|-----------|-------------|---------|
| BF16 bs=192 | 24 | ~25 min | ~76 min | 4 处 dtype patch |
| FP32 bs=128 | 16 | ~26 min | ~79 min | 无 |
| H200 BF16 bs=384 (参考) | 48 | ~39 min | ~117 min* | 1 处 dtype patch |

*H200 数据为 25 epoch 训练的 per-epoch 平均值

## CUDA Kernel Profile 分析

**环境**: 单卡 L40S, FP32, bs=16, 15 步 profile

### GPU 时间分布 (Self CUDA Time)

| 操作 | CUDA 时间 | 占比 | 说明 |
|------|----------|------|------|
| **upsample_linear1d (forward)** | **116.4 ms** | **28.6%** | F.interpolate(mode="linear") |
| **upsample_linear1d_backward** | **142.3 ms** | **35.0%** | linear 插值的反向传播 |
| **F.interpolate 总计** | **258.7 ms** | **63.6%** | ⚠️ 压倒性瓶颈 |
| AdamW optimizer | 87.4 ms | 21.5% | 参数更新 |
| mm (矩阵乘法) | 30.9 ms | 7.6% | Transformer attention/FFN |
| addmm | 22.9 ms | 5.6% | Linear 层 |
| sum | 10.1 ms | 2.5% | |
| layer_norm backward | 7.6 ms | 1.9% | |
| copy_ | 7.8 ms | 1.9% | |
| convolution backward | 5.3 ms | 1.3% | Conv1d |
| bmm | 5.5 ms | 1.4% | Batch matmul |
| NATTEN (NA1D) | 4.8 ms | 1.2% | Neighborhood Attention |

### 关键发现

**`F.interpolate(mode="linear")` 占了 63.6% 的 GPU 计算时间。**

这是 Pluto FPN (Feature Pyramid Network) 中的 1D 时间维度上采样操作：
- 位置: `src/models/pluto/layers/embedding.py:78`
- 两次上采样: 6→11 (scale 1.83) 和 11→21 (scale 1.91)
- 每步仅 24 次调用，但每次耗时极长

### 瓶颈根因

1. `F.interpolate(mode="linear")` 的 CUDA kernel 效率极低：
   - 不走 Tensor Core
   - 大量小 tensor 操作导致 kernel launch overhead 高
   - backward 比 forward 更慢 (142ms vs 116ms)

2. 模型本身很小 (4.1M 参数)：
   - Transformer 的 matmul 只占 13.2%
   - 大部分 GPU 算力在等 linear interpolation

### torch.compile 测试 (PyTorch 2.5.0+cu124)

使用 `torch.compile(model, mode="default")` 对比：

| 指标 | 无 compile | 有 compile | 变化 |
|------|-----------|-----------|------|
| upsample_linear1d forward | 116.4ms (28.6%) | 116.4ms (30.3%) | 不变 |
| upsample_linear1d backward | 142.3ms (35.0%) | 142.3ms (37.0%) | 不变 |
| interpolate 总计 | 258.7ms (63.6%) | 258.8ms (67.2%) | **不变** |
| Self CUDA total | 407.1ms | 384.8ms | 略快 5% |

**结论**: `torch.compile` 优化了 elementwise 等小操作，但 **`F.interpolate` 的 CUDA kernel 无法被 Inductor 优化** — 它是独立的 C++ kernel，不在融合范围内。NATTEN 也会导致 graph break（`na1d_qk_forward` / `na1d_av_forward` 不被 dynamo 支持）。

此外 `mode="reduce-overhead"` 因使用 CUDA Graphs 与重复 batch 的 backward 冲突，无法使用。

### 优化建议

| 方案 | 预期效果 | 改动量 | 状态 |
|------|---------|--------|------|
| 改 `mode="nearest"` | 省 ~60% GPU 时间 | 改 1 行 | 待测试 |
| 改 learnable ConvTranspose1d | 省 ~60% + 走 Tensor Core + 可学习 | 改 ~15 行 | 待测试 |
| `torch.compile` | 省 ~5% (仅小操作) | 加 1 行 | ❌ 对瓶颈无效 |
| 禁用 RichProgressBar | 省 ~15% 训练时间 (PL overhead) | 加 1 行配置 | 待测试 |

## Feature Cache 阶段

| 指标 | 值 |
|------|-----|
| 总 scenario | 31,250 |
| Ray workers | 32 |
| 总耗时 | ~5 小时 |
| Cache 大小 | 485 GB |
| 速度 | ~6,000 scenario/hr (稳定期) |

## 环境搭建问题与解决方案

### 1. DLAMI 无 conda
DLAMI 使用系统 Python 3.10，改用 `python3 -m venv` 创建虚拟环境。

### 2. numpy 版本冲突
nuplan-devkit 使用 `np.bool`（1.24+ 移除），降级到 numpy 1.23.5。

### 3. torchmetrics 不兼容
pluto 使用 `compute_on_step` 参数（新版移除），降级到 torchmetrics 0.11.4。

### 4. pkg_resources 缺失
setuptools 82+ 移除了 pkg_resources，降级到 `setuptools<82`。

### 5. NCCL OFI 插件失败
g6e 无 EFA，`aws-ofi-nccl` 初始化失败。设 `NCCL_NET=Socket` 跳过。

### 6. Ray workers GPU 不可见
Ray cache workers 启动时 `num_gpus=0`，NATTEN import 检测 GPU 失败。设 `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`。

### 7. BF16 dtype 不匹配 (4 处)
`torch.zeros()` 默认创建 FP32 tensor，与 BF16 的 `index_put` 冲突：
- `agent_encoder.py:83` — 加 `dtype=x_agent_tmp.dtype`
- `embedding.py:283` — 加 `dtype=x_valid.dtype`
- `embedding.py:292` — 加 `dtype=x_features_valid.dtype`
- `map_encoder.py:85` — 加 `dtype=x_polygon.dtype`
- `map_encoder.py:89` — `self.unknown_speed_emb.weight.to(x_speed_limit.dtype)`

### 8. RichProgressBar 崩溃
非终端模式下 `_live_stack.pop()` 报错，patch `rich/console.py` safe pop。

### 9. custom_training/__init__.py 为空
手动添加 `TrainingEngine`, `build_training_engine`, `update_config_for_training` imports。

### 10. 数据目录结构不匹配
nuplan 期望 `/nuplan/dataset/nuplan-v1.1/trainval/`，实际数据在 `/nuplan/dataset/data/cache/`。创建 8734 个 .db 符号链接。

### 11. pluto.pth (Python path)
Ray workers 无法 import `src.models`，在 venv site-packages 添加 `/opt/dlami/nvme/pluto` 的 .pth 文件。

## G6e vs H200 对比

| 指标 | H200 (p5en.48xlarge) | L40S (g6e.48xlarge) |
|------|---------------------|---------------------|
| GPU | 8x H200 144GB HBM3e | 8x L40S 45GB GDDR6 |
| BF16 TFLOPS | 1,979 | 362 (18%) |
| 显存带宽 | 4.8 TB/s | 864 GB/s (18%) |
| GPU 互联 | NVLink 900 GB/s | PCIe ~25 GB/s |
| 价格 | ~$999/24h (CB) | ~$724/24h (OD) |
| Feature Cache | ~2.5 hr | ~5 hr |
| 每 Epoch (BF16) | ~39 min (bs=384) | ~25 min (bs=192) |
| GPU 利用率 | 75% avg | 85-100% |
| 显存使用 | 18-29 GB (12-20%) | 38-45 GB (85-100%) |
| 功耗 | ~145W (20% TDP) | 100-195W (28-56% TDP) |

**注意**: L40S 每 epoch 更快是因为 batch size 更小导致步数更多但每步更快。CUDA profile 显示 63.6% 的 GPU 时间花在 `F.interpolate(mode="linear")` 上（FPN 的 1D 时间维度上采样：6→11 和 11→21），而非 Transformer 计算。`torch.compile` 无法优化该 kernel，需要改用 `mode="nearest"` 或 learnable ConvTranspose1d 来消除瓶颈。
