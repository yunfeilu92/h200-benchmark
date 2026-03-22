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

L40S 每卡 48GB，但 Pluto 模型很小（H200 上仅用 18-29GB），48GB 完全足够。
batch_size 保持 384 (48/卡) 不变。

### 3. 计算性能

| 指标 | H200 | L40S | 比率 |
|------|------|------|------|
| BF16 TFLOPS | 1,979 | 362 | L40S = 18% |
| FP32 TFLOPS | 67 | 91 | L40S 略高 |
| 显存带宽 | 4.8 TB/s (HBM3e) | 864 GB/s (GDDR6) | L40S = 18% |

预期训练速度：L40S 约为 H200 的 20-30%（受 BF16 算力和显存带宽限制）。

### 4. 网络

g6e 无 EFA，使用 100 Gbps ENA。单节点 8 卡训练不受影响（GPU 间通信走 PCIe）。

## 性能调优 checklist

- [x] bf16-mixed 精度（BF16 算力远高于 FP32）
- [x] NCCL Ring 算法 + PCIe P2P 优化
- [x] CUDA_DEVICE_MAX_CONNECTIONS=1（compute/communication overlap）
- [x] PyTorch CUDA 内存分配器优化
- [x] NVMe SSD 用于数据和 cache
- [ ] torch.compile（需验证 Pluto 兼容性，可能需要修改代码）
- [ ] Flash Attention（Pluto 使用 NATTEN，不直接适用）
- [ ] 更大 batch size（如显存允许，可尝试 512 即 64/卡）
- [ ] gradient accumulation（如 PCIe 通信是瓶颈，可累积梯度减少 all-reduce 频率）

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

## 预期结果

基于 H200 benchmark 数据外推：
- Feature Cache: ~3-4 小时（H200 约 2.5 小时，受 CPU/IO 限制差异不大）
- 每 Epoch 训练: ~120-180 分钟（H200 约 39 分钟，受 BF16 算力差距影响）
- 总训练 25 Epoch: ~50-75 小时

注意：以上为粗略估算，实际性能取决于 PCIe 通信开销和模型特性。
