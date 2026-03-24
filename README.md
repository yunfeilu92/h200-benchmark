# GPU Benchmark: Pluto (Autonomous Driving Planning)

Benchmark [Pluto](https://github.com/jchengai/pluto) (4.1M param Transformer planning model) on AWS GPU instances using the [nuPlan](https://www.nuscenes.org/nuplan) dataset.

## Key Finding: `F.interpolate(mode="linear")` is the bottleneck

CUDA kernel profiling revealed that **63.6% of GPU time** was spent on `F.interpolate(mode="linear")` in the FPN's 1D temporal upsampling (6->11->21 frames), not on Transformer computation.

Simply changing to `mode="nearest"` (1 line change) achieved:
- **47% faster training** (79 min -> 42 min for 3 epochs)
- **2.4% lower val_loss** (3.617 vs 3.706)
- Zero additional parameters

## Results Summary

### Interpolation Mode Comparison (g6e, FP32, bs=128, 3 epochs)

| Config | Training Time | val_loss | Code Change |
|--------|-------------|----------|-------------|
| FP32 + linear (original) | 79 min | 3.706 | none |
| BF16 + linear | 76 min | 4.077 | 4 dtype patches |
| FP32 + ConvTranspose1d | 42 min | 3.783 | ~15 lines |
| **FP32 + nearest** | **42 min** | **3.617** | **1 line** |

### GPU Instance Comparison

| Instance | GPU | Training/Epoch | Price |
|----------|-----|---------------|-------|
| p5en.48xlarge | 8x H200 144GB | ~39 min (BF16, bs=384) | ~$999/24h |
| g6e.48xlarge | 8x L40S 45GB | ~14 min (FP32+nearest, bs=128) | ~$724/24h |

### CUDA Profile (Self CUDA Time, single L40S)

| Mode | upsample fwd | upsample bwd | Total GPU | interpolate % |
|------|-------------|-------------|-----------|---------------|
| **linear** | 116.4ms | 142.3ms | 407.1ms | **63.6%** |
| **ConvTranspose1d** | ~0ms | ~1ms | 150.1ms | **~0%** |

### Optimization Attempts

| Approach | Effect | Status |
|----------|--------|--------|
| `mode="nearest"` | -47% time, -2.4% loss | Best |
| Learnable ConvTranspose1d | -47% time, +2.1% loss | Good |
| `torch.compile` | -5% (small ops only) | Ineffective on bottleneck |
| BF16 precision | -4% time, +10% loss | Not recommended for this model |

## Reports

- [G6e Benchmark Report](G6E-BENCHMARK-REPORT.md) - Full g6e (L40S) benchmark with CUDA profiling
- [H200 Benchmark Report](H200-BENCHMARK-REPORT.md) - H200 baseline benchmark
- [G6e Setup Guide](G6E-SETUP-GUIDE.md) - Environment configuration for g6e
- [H200 Capacity Blocks Guide](H200-CAPACITY-BLOCKS-GUIDE.md) - AWS Capacity Blocks purchasing guide

## Scripts

| Script | Description |
|--------|-------------|
| `setup.sh` | H200 environment setup |
| `setup_g6e.sh` | G6e environment setup (venv, no conda) |
| `run_benchmark_v2.sh` | H200 training benchmark |
| `run_benchmark_g6e.sh` | G6e training benchmark |
| `download_data.sh` | nuPlan dataset download |

## Environment Issues & Solutions

11 environment issues were encountered and solved during g6e setup, including numpy version conflicts, NCCL OFI plugin failures (no EFA), BF16 dtype mismatches, and Ray worker GPU visibility. See [G6e Benchmark Report](G6E-BENCHMARK-REPORT.md#environment-issues) for details.
