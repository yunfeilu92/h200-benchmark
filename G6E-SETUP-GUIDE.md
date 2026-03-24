# G6e (8x L40S) Benchmark 配置指南

## 实例规格

| 项目 | g6e.48xlarge |
|------|-------------|
| GPU | 8x NVIDIA L40S |
| GPU 显存 | 48 GB GDDR6 / 卡 |
| GPU 架构 | Ada Lovelace (SM 8.9) |
| vCPU | 192 |
| 内存 | 768 GB |
| 本地存储 | 7.6 TB NVMe SSD |
| 网络 | 100 Gbps ENA (无 EFA) |
| GPU 互联 | PCIe Gen4 x16 (无 NVLink) |
| GPU TDP | 350W / 卡 |

## 最优软件栈

| 组件 | 版本 | 说明 |
|------|------|------|
| AMI | Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04) | 预装 NVIDIA 驱动 + CUDA |
| NVIDIA Driver | 550.x+ | DLAMI 自带，Ada Lovelace 最佳支持 |
| CUDA | 12.4 | DLAMI 原生，sm_89 完整支持 |
| PyTorch | 2.5.0+cu124 | torch.compile 对 Ada Lovelace 优化好 |
| NCCL | 2.21.5+ (PyTorch 自带) | 需额外环境变量调优 PCIe 拓扑 |
| NATTEN | 0.17.5+torch250cu124 | Pluto 需要的 Neighborhood Attention |
| Python | 3.10 | nuplan-devkit 兼容性 |
| Precision | bf16-mixed | L40S BF16 算力远超 FP32 |

## 与 H200 的关键差异

### 1. GPU 互联：PCIe vs NVLink

**这是最大的差异。** H200 (p5en) 8 卡通过 NVSwitch 全互联，带宽 900 GB/s。G6e 的 L40S 通过 PCIe Gen4 互联，带宽仅 ~25 GB/s。

影响：
- DDP all-reduce 通信开销显著增加
- 梯度同步时间占比更高
- 不适合 tensor parallelism

NCCL 优化配置：
```bash
export NCCL_ALGO=Ring            # Ring 在 PCIe 拓扑下最优
export NCCL_PROTO=Simple         # 降低协议开销
export NCCL_P2P_LEVEL=PHB        # PCIe Host Bridge 级别
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=12
export NCCL_BUFFSIZE=8388608     # 8MB buffer
```

### 2. 显存：48GB vs 144GB

L40S 每卡 45GB 可用，FP32 bs=128 (16/卡) 使用 31-45GB。

### 3. 计算性能

| 指标 | H200 | L40S | 比率 |
|------|------|------|------|
| BF16 TFLOPS | 1,979 | 362 | L40S = 18% |
| FP32 TFLOPS | 67 | 91 | L40S 略高 |
| 显存带宽 | 4.8 TB/s (HBM3e) | 864 GB/s (GDDR6) | L40S = 18% |

**实测结果**: L40S FP32+nearest 每 Epoch ~14 min，比 H200 BF16+linear ~39 min 更快（Pluto 模型太小，H200 算力严重过剩）。

### 4. 网络

g6e 无 EFA，使用 400 Gbps ENA。单节点 8 卡训练不受影响（GPU 间通信走 PCIe）。需设 `NCCL_NET=Socket` 禁用 OFI 插件。

## 性能调优 checklist

- [x] **F.interpolate mode="nearest"**（关键优化！linear 占 63.6% GPU 时间，改 nearest 后训练加速 47%）
- [x] FP32 精度（BF16 在此模型无优势且 loss 更差）
- [x] NCCL Ring 算法 + PCIe P2P + Socket 网络
- [x] CUDA_DEVICE_MAX_CONNECTIONS=1
- [x] NVMe SSD 用于数据和 cache
- [x] NCCL_NET=Socket（禁用 OFI/EFA 插件）
- [x] RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0（Ray worker GPU 可见性）
- [x] torch.compile — 已测试，对 F.interpolate 瓶颈无效
- [ ] 禁用 RichProgressBar（可再省 ~15%）

## 推荐 AMI 查询

```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=*Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 22.04)*" \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
  --output text \
  --region us-east-1
```

## 运行步骤

```bash
# 1. 启动 g6e.48xlarge (选择上述 AMI)
# 2. SSH 到实例
ssh -i your-key.pem ubuntu@<public-ip>

# 3. 上传脚本
scp -i your-key.pem setup_g6e.sh run_benchmark_g6e.sh download_data.sh ubuntu@<public-ip>:/home/ubuntu/

# 4. 环境搭建
bash setup_g6e.sh

# 5. 下载数据
bash download_data.sh

# 6. 运行 benchmark
nohup bash run_benchmark_g6e.sh > benchmark_output.log 2>&1 &

# 7. 监控
tail -f /home/ubuntu/train_g6e.log
watch -n 5 nvidia-smi
```

## 实测结果

| 阶段 | 耗时 |
|------|------|
| Feature Cache (32 Ray workers) | ~5 小时 |
| 每 Epoch (FP32 + nearest, bs=128) | ~14 min |
| 25 Epoch 训练预计 | ~6 小时 |
| 总计（含 cache） | ~11 小时 |

## Pluto 代码必要修改

在运行 benchmark 前需要修改 Pluto 代码：

```bash
# 1. embedding.py: F.interpolate mode 改为 nearest（关键性能优化）
# src/models/pluto/layers/embedding.py:81
#   mode="linear" → mode="nearest"
#   删除 align_corners=False 行

# 2. agent_encoder.py:83: 加 dtype（FP32 下非必须，BF16 需要）
# 3. custom_training/__init__.py: 加 TrainingEngine 等 import
# 4. 创建 pluto.pth: echo "/opt/dlami/nvme/pluto" > $(python -c "import site; print(site.getsitepackages()[0])")/pluto.pth
```

详见 [G6E-BENCHMARK-REPORT.md](G6E-BENCHMARK-REPORT.md) 完整修改列表。
