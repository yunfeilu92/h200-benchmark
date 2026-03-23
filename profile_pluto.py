import torch
from torch.profiler import profile, ProfilerActivity
import os, sys, time

os.environ["NUPLAN_DATA_ROOT"] = "/nuplan/dataset"
os.environ["NUPLAN_MAPS_ROOT"] = "/nuplan/dataset/maps/nuplan-maps-v1.0"
os.environ["NUPLAN_EXP_ROOT"] = "/nuplan/exp"

sys.path.insert(0, "/opt/dlami/nvme/pluto")

from src.models.pluto.pluto_model import PlanningModel

# Build model with exact config
model = PlanningModel(
    dim=128, state_channel=6, polygon_channel=6, history_channel=9,
    history_steps=21, future_steps=80, encoder_depth=4, decoder_depth=4,
    drop_path=0.2, dropout=0.1, num_heads=4, num_modes=12,
    state_dropout=0.75, use_ego_history=False, state_attn_encoder=True,
    use_hidden_proj=False,
).cuda().bfloat16()
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params:,} params ({total_params*2/1024/1024:.1f} MB BF16)")

# Load a real batch from cache using PlutoFeature
from src.features.pluto_feature import PlutoFeature
import gzip, pickle, numpy as np

cache_dir = "/nuplan/exp/cache_pluto"
sample_dirs = []
for log_dir in sorted(os.listdir(cache_dir))[:20]:
    log_path = os.path.join(cache_dir, log_dir)
    if not os.path.isdir(log_path):
        continue
    for scenario_type in os.listdir(log_path):
        st_path = os.path.join(log_path, scenario_type)
        if not os.path.isdir(st_path):
            continue
        for token in os.listdir(st_path):
            sample_dirs.append(os.path.join(st_path, token))
            if len(sample_dirs) >= 384:
                break
        if len(sample_dirs) >= 384:
            break
    if len(sample_dirs) >= 384:
        break

print(f"Loading {len(sample_dirs)} samples from cache...")
features = []
for d in sample_dirs:
    feat_path = os.path.join(d, "feature.gz")
    with gzip.open(feat_path, "rb") as f:
        raw = pickle.load(f)
    # Convert numpy arrays to tensors recursively
    def np_to_tensor(obj):
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        elif isinstance(obj, dict):
            return {k: np_to_tensor(v) for k, v in obj.items()}
        elif isinstance(obj, (float, int, np.floating, np.integer)):
            return torch.tensor(obj)
        return obj
    features.append(PlutoFeature(np_to_tensor(raw["data"])))

# Collate
data = PlutoFeature.collate(features)

# Move to GPU
def to_cuda(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cuda()
    elif isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cuda(v) for v in obj]
    return obj

def to_float32(obj):
    if isinstance(obj, torch.Tensor) and obj.dtype == torch.float64:
        return obj.float()
    elif isinstance(obj, dict):
        return {k: to_float32(v) for k, v in obj.items()}
    return obj

data = to_float32(to_cuda(data.data))
bs = data["agent"]["position"].shape[0]
print(f"Batch on GPU, bs={bs}")

# Warmup
for _ in range(3):
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(data)
torch.cuda.synchronize()

# Measure wall time
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 50
for _ in range(N):
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(data)
torch.cuda.synchronize()
t1 = time.perf_counter()
wall_per_step = (t1 - t0) / N * 1000
print(f"\nWall time per forward: {wall_per_step:.1f} ms")
print(f"Throughput: {bs / (wall_per_step/1000):.0f} samples/sec")

# Profile with FLOPS counting
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
    for _ in range(10):
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(data)
    torch.cuda.synchronize()

print("\n=== TOP 20 CUDA KERNELS BY GPU TIME ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Compute utilization
events = prof.key_averages()
total_flops = sum(e.flops for e in events if e.flops > 0)
total_cuda_us = sum(e.cuda_time_total for e in events)
matmul_time = sum(e.cuda_time_total for e in events if "mm" in e.key.lower() or "gemm" in e.key.lower() or "addmm" in e.key.lower())
attn_time = sum(e.cuda_time_total for e in events if "attn" in e.key.lower() or "attention" in e.key.lower() or "sdp" in e.key.lower() or "flash" in e.key.lower())
elementwise_time = sum(e.cuda_time_total for e in events if any(x in e.key.lower() for x in ["add", "mul", "relu", "gelu", "sigmoid", "tanh", "norm", "layer_norm", "batch_norm"]))
copy_index_time = sum(e.cuda_time_total for e in events if any(x in e.key.lower() for x in ["copy", "index", "scatter", "gather", "cat", "stack", "where", "masked"]))

print(f"\n{'='*60}")
print(f"=== H200 UTILIZATION ANALYSIS (10 forward, bs={bs}) ===")
print(f"{'='*60}")
print(f"\n--- Time Breakdown ---")
print(f"Total CUDA time:     {total_cuda_us/1000:.1f} ms")
print(f"MatMul/GEMM time:    {matmul_time/1000:.1f} ms ({matmul_time/total_cuda_us*100:.1f}%)")
print(f"Attention time:      {attn_time/1000:.1f} ms ({attn_time/total_cuda_us*100:.1f}%)")
print(f"Elementwise time:    {elementwise_time/1000:.1f} ms ({elementwise_time/total_cuda_us*100:.1f}%)")
print(f"Copy/Index time:     {copy_index_time/1000:.1f} ms ({copy_index_time/total_cuda_us*100:.1f}%)")
other = total_cuda_us - matmul_time - attn_time - elementwise_time - copy_index_time
print(f"Other time:          {other/1000:.1f} ms ({other/total_cuda_us*100:.1f}%)")

print(f"\n--- Compute Utilization ---")
print(f"Total measured FLOPS: {total_flops:.3e}")
if total_cuda_us > 0 and total_flops > 0:
    achieved = total_flops / (total_cuda_us * 1e-6) / 1e12
    print(f"Achieved throughput:  {achieved:.2f} TFLOPS")
    print(f"H200 BF16 peak:       1979 TFLOPS")
    print(f"Compute utilization:  {achieved/1979*100:.2f}%")

print(f"\n--- Why H200 is underutilized ---")
print(f"1. Model size: {total_params:,} params = {total_params*2/1024/1024:.1f} MB (tiny)")
print(f"2. GEMM dimensions: {128}x{128} (dim=128), too small for 528 Tensor Cores")
print(f"3. MatMul accounts for only {matmul_time/total_cuda_us*100:.1f}% of GPU time")
print(f"4. Kernel count: {sum(1 for e in events if e.cuda_time > 0)} unique kernels in one forward pass")
