# AWS H200 GPU Capacity Blocks 实战指南

> 调研日期: 2026-03-22

## 1. H200 实例类型总览

| 实例类型 | GPU | GPU 数量 | GPU Memory | vCPU | RAM | 网络 | us-east-1 CB 支持 |
|----------|-----|---------|------------|------|-----|------|-------------------|
| **p5e.48xlarge** | H200 (HBM3e) | 8 | 8x141GB = 1128GB | 192 | 2048 GiB | 3200 Gbps EFAv2 | **不支持** |
| **p5en.48xlarge** | H200 (HBM3e) | 8 | 8x141GB = 1128GB | 192 | 2048 GiB | 3200 Gbps EFAv3 | **支持** (当前无库存) |
| **p6-b200.48xlarge** | B200 | 8 | TBD | TBD | TBD | TBD | **支持** (有库存) |

### p5e vs p5en 区别
- **p5en** 比 p5e 多了 EFAv3（第三代 Elastic Fabric Adapter）+ Nitro v5，网络延迟降低 35%
- **p5en** 使用 Gen5 PCIe（CPU-GPU 带宽提升 4x）
- p5en 是 p5e 的升级版，推荐优先选择 p5en

### Capacity Block 支持的 Region（H200 相关）

**p5e.48xlarge**: us-east-2, us-west-1, us-west-2, eu-north-1, eu-west-2, eu-south-2, sa-east-1, ap-northeast-1/2, ap-south-1, ap-southeast-3
- **注意：p5e 不在 us-east-1 支持列表中**

**p5en.48xlarge**: us-east-1, us-east-2, us-west-2, ap-northeast-1 (文档确认，但当前 us-east-1 库存为空)

## 2. Capacity Block 购买流程 (CLI)

### Step 1: 查询可用 Capacity Block

```bash
# 查询 p5en.48xlarge（H200）在 us-east-1，1天
aws ec2 describe-capacity-block-offerings \
  --instance-type p5en.48xlarge \
  --instance-count 1 \
  --capacity-duration-hours 24 \
  --region us-east-1

# 查询 p5e.48xlarge 在 us-east-2，7天
aws ec2 describe-capacity-block-offerings \
  --instance-type p5e.48xlarge \
  --instance-count 1 \
  --capacity-duration-hours 168 \
  --region us-east-2

# 指定日期范围查询
aws ec2 describe-capacity-block-offerings \
  --instance-type p5en.48xlarge \
  --instance-count 1 \
  --capacity-duration-hours 24 \
  --start-date-range 2026-03-25T00:00:00Z \
  --end-date-range 2026-04-01T00:00:00Z \
  --region us-east-1
```

**Duration 规则**: 1-14 天（按天递增），或 7 天的倍数直到 182 天

### Step 2: 购买 Capacity Block

```bash
# 使用上一步返回的 CapacityBlockOfferingId
aws ec2 purchase-capacity-block \
  --capacity-block-offering-id cb-0123456789abcdefg \
  --instance-platform Linux/UNIX \
  --region us-east-1
```

购买后状态变化: `payment-pending` -> `scheduled` -> `active`

### Step 3: 查看 AMI ID（Deep Learning AMI）

```bash
# 推荐：PyTorch 2.9 + Ubuntu 24.04（最新，支持 P5/P5e/P5en）
aws ssm get-parameter \
  --region us-east-1 \
  --name "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.9-ubuntu-24.04/latest/ami-id" \
  --query "Parameter.Value" \
  --output text
# 当前返回: ami-08beb2510fdf8bd81

# 备选：PyTorch 2.7 + Ubuntu 22.04
aws ssm get-parameter \
  --region us-east-1 \
  --name "/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id" \
  --query "Parameter.Value" \
  --output text
# 当前返回: ami-084a4fb2032f7c35c

# 列出所有可用的 PyTorch DLAMI
aws ssm get-parameters-by-path \
  --region us-east-1 \
  --path "/aws/service/deeplearning/ami/" \
  --recursive \
  --query "Parameters[*].Name" \
  --output json | python3 -c "
import json, sys
names = json.load(sys.stdin)
for n in sorted(names):
    if 'pytorch' in n.lower() and 'oss-nvidia' in n:
        print(n)
"
```

### Step 4: 启动实例到 Capacity Block

```bash
# 获取 Capacity Reservation ID（从 purchase 的输出中获取，或查询）
aws ec2 describe-capacity-reservations \
  --filters Name=instance-type,Values=p5en.48xlarge \
  --region us-east-1 \
  --query "CapacityReservations[?ReservationType=='capacity-block']"

# 启动实例
aws ec2 run-instances \
  --image-id ami-08beb2510fdf8bd81 \
  --instance-type p5en.48xlarge \
  --count 1 \
  --key-name YOUR_KEY_PAIR \
  --instance-market-options '{"MarketType":"capacity-block"}' \
  --capacity-reservation-specification '{"CapacityReservationTarget":{"CapacityReservationId":"cr-0123456789abcdefg"}}' \
  --region us-east-1
```

## 3. 前提条件 & 注意事项

### Service Quotas
- **Capacity Block 不计入 On-Demand 配额**，无需提前申请 On-Demand quota
- 每个 Capacity Block 最多 64 个实例
- 跨所有 Capacity Block（同一 AWS Organization）最多 256 个实例
- 购买后**无法取消**

### 时间相关
- Capacity Block 统一在 **11:30 AM UTC** 结束
- 终止流程在最后一天 **11:00 AM UTC** 开始（提前 30 分钟）
- 最多提前 8 周预订
- 最短可提前 30 分钟开始

### 网络和 EFA
- 使用自定义 AMI 时，需确保已安装 EFA 驱动和 NCCL 插件
- DLAMI 已预装所有必要的 EFA/NCCL 组件
- P5en 实例需要配置 NetworkCardIndex 和 DeviceIndex

### 安全组
- 需要开放 EFA 通信端口（如多节点训练）
- 建议创建专用 placement group

## 4. 定价信息（2026年1月更新后，含15%涨幅）

| 实例类型 | Region | 每小时每实例 | 每小时每GPU | 24小时总费用 |
|----------|--------|------------|------------|-------------|
| p5e.48xlarge | us-east-2 | $39.799 | $4.975 | ~$955 |
| p5e.48xlarge | us-west-1 | $49.749 | $6.219 | ~$1,194 |
| p5en.48xlarge | us-east-1 | $41.612 | $5.201 | ~$999 |
| p5en.48xlarge | us-east-2 | $41.612 | $5.201 | ~$999 |
| p5en.48xlarge | us-west-2 | $41.612 | $5.201 | ~$999 |
| p6-b200.48xlarge | us-east-1 | $74.88 | $9.36 | ~$1,797 |

> 价格为动态定价，预计 2026年4月 再次更新。实际费用以 `describe-capacity-block-offerings` 返回的 `UpfrontFee` 为准。

## 5. 推荐方案（GPU Training Benchmark）

### 首选：p5en.48xlarge in us-east-1
- H200 GPU x8，EFAv3 网络
- 如果 us-east-1 无库存，fallback 到 us-east-2 或 us-west-2

### 备选：p5e.48xlarge in us-east-2
- H200 GPU x8，EFAv2 网络
- p5e 不支持 us-east-1

### AMI 推荐
- **PyTorch 2.9 + Ubuntu 24.04** (OSS NVIDIA Driver): `ami-08beb2510fdf8bd81` (us-east-1)
- 包含: CUDA 12.x, NCCL, EFA, PyTorch 2.9

## 6. 快速脚本：查询并购买

```bash
#!/bin/bash
# h200-cb-purchase.sh
# 查询并购买 H200 Capacity Block

REGION="us-east-1"
INSTANCE_TYPE="p5en.48xlarge"
INSTANCE_COUNT=1
DURATION_HOURS=24  # 1 day

echo "Querying capacity block offerings..."
OFFERINGS=$(aws ec2 describe-capacity-block-offerings \
  --instance-type $INSTANCE_TYPE \
  --instance-count $INSTANCE_COUNT \
  --capacity-duration-hours $DURATION_HOURS \
  --region $REGION \
  --output json)

echo "$OFFERINGS" | python3 -c "
import json, sys
data = json.load(sys.stdin)
offerings = data.get('CapacityBlockOfferings', [])
if not offerings:
    print('No offerings available. Try a different region or date range.')
    sys.exit(1)
for i, o in enumerate(offerings):
    print(f\"[{i}] ID: {o['CapacityBlockOfferingId']}\")
    print(f\"    AZ: {o['AvailabilityZone']}\")
    print(f\"    Start: {o['StartDate']}\")
    print(f\"    End: {o['EndDate']}\")
    print(f\"    Duration: {o['CapacityBlockDurationHours']}h\")
    print(f\"    Price: \${o['UpfrontFee']} {o['CurrencyCode']}\")
    print()
"

# 取消下面的注释来购买（需要替换 OFFERING_ID）
# OFFERING_ID="cb-xxxx"
# aws ec2 purchase-capacity-block \
#   --capacity-block-offering-id $OFFERING_ID \
#   --instance-platform Linux/UNIX \
#   --region $REGION
```

## 7. 实时查询结果快照 (2026-03-22)

### us-east-1
- p5e.48xlarge: **不支持此 region**
- p5en.48xlarge: 支持但当前**无库存**
- p6-b200.48xlarge: **有库存** ($1,797/24h)

### us-east-2
- p5en.48xlarge: **有库存** ($999/24h) - **性价比最优**

### us-west-2
- p5en.48xlarge: **有库存** ($999/24h)

---

## Sources
- [EC2 Capacity Blocks 官方文档](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html)
- [Capacity Blocks 购买指南](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-purchase.html)
- [Capacity Blocks 定价](https://aws.amazon.com/ec2/capacityblocks/pricing/)
- [EC2 P5 实例类型](https://aws.amazon.com/ec2/instance-types/p5/)
- [P5en 发布公告](https://aws.amazon.com/blogs/aws/new-amazon-ec2-p5en-instances-with-nvidia-h200-tensor-core-gpus-and-efav3-networking/)
- [P5e Capacity Blocks 发布公告](https://aws.amazon.com/about-aws/whats-new/2024/09/amazon-ec2-p5e-instances-ec2-capacity-blocks/)
- [AWS DLAMI 文档](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html)
