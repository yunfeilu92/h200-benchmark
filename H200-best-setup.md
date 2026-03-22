 H200 (p5en.48xlarge on AWS) 最佳配置

    版本调研结论：

    - CUDA 12.4（H200 Hopper 架构 sm_90a，DLAMI 自带 12.4 已最优，无需升级到 12.8，12.8 是 Blackwell sm_120 的要求）
    - PyTorch 2.5.1 cu124（DLAMI 原生版本，完整支持 sm_90a + BF16 所有 ops + torch.compile）
    - NCCL 2.21.5（PyTorch 2.5 自带，p5en 配 EFAv3 网络，NCCL 原生支持 EFA，无需额外配置）
    - Python 3.12+（3.9 已 EOL，3.12 有显著性能提升，特别是 GIL 优化和 DataLoader 效率）
    - precision: bf16-mixed（H200 BF16 算力 1979 TFLOPS vs FP32 67 TFLOPS，差 30 倍）
    - DDP 在 p5en 上正常工作，不需要 NCCL_P2P_DISABLE=1（那是 PCIe 拓扑问题，p5en 用 NVSwitch 全互联，P2P 直接走 NVLink，无此问题）

    额外优化建议：

    - NCCL_ALGO=Ring（8 GPU NVSwitch 拓扑下 Ring 通常最优）
    - torch.compile(mode="reduce-overhead")（减少 kernel launch 开销，对小模型特别有效）
    - CUDA_DEVICE_MAX_CONNECTIONS=1（改善 compute/communication overlap）
    - TF32 自动启用（PyTorch 2.5 默认开启 torch.backends.cuda.matmul.allow_tf32=True，FP32 运算自动用 TF32，989 TFLOPS）
    - DataLoader: pin_memory=True, persistent_workers=True, prefetch_factor=4
    - EFA 网络：p5en 的 EFAv3 提供 3.2Tbps 网络带宽，多节点训练时设 FI_EFA_USE_DEVICE_RDMA=1

    与你当前环境的差异：

    ┌───────────┬────────────────────────────┬───────────────────────────┐
    │   项目    │            当前            │           推荐            │
    ├───────────┼────────────────────────────┼───────────────────────────┤
    │ PyTorch   │ 2.0.1+cu118 → 正在升级 2.5 │ 2.5.1+cu124               │
    ├───────────┼────────────────────────────┼───────────────────────────┤
    │ CUDA      │ 11.8 (PyTorch)             │ 12.4 (DLAMI 原生)         │
    ├───────────┼────────────────────────────┼───────────────────────────┤
    │ Python    │ 3.9                        │ 3.12                      │
    ├───────────┼────────────────────────────┼───────────────────────────┤
    │ Precision │ FP32                       │ bf16-mixed                │
    ├───────────┼────────────────────────────┼───────────────────────────┤
    │ NATTEN    │ 0.14.6                     │ 需匹配 torch 2.5 版本     │
    ├───────────┼────────────────────────────┼───────────────────────────┤
    │ NCCL P2P  │ 默认开启                   │ 保持开启（NVSwitch 直连） │
    └───────────┴────────────────────────────┴───────────────────────────┘

    注意事项：

    - H200 是 Hopper 架构（sm_90a），不是 Blackwell（sm_120），不需要 CUDA 12.8
    - AWS p5en 用 NVSwitch 互联 8 块 H200，GPU 间带宽 900 GB/s，DDP 通信不是瓶颈
    - 功耗上限 700W/卡，如果实际功耗 < 300W 说明模型太小没喂饱 GPU、




    当前配置

    环境（pluto310）

    ┌───────────────┬──────────────────────┐
    │     项目      │          值          │
    ├───────────────┼──────────────────────┤
    │ Python        │ 3.10                 │
    ├───────────────┼──────────────────────┤
    │ PyTorch       │ 2.5.0+cu124          │
    ├───────────────┼──────────────────────┤
    │ CUDA          │ 12.4                 │
    ├───────────────┼──────────────────────┤
    │ NATTEN        │ 0.17.5+torch250cu124 │
    ├───────────────┼──────────────────────┤
    │ nuplan-devkit │ 1.2.2                │
    └───────────────┴──────────────────────┘

    训练精度

    bf16-mixed（BF16 混合精度）

    训练参数

    - batch_size=384（48/卡）
    - 8x H200 DDP
    - NCCL_ALGO=Ring
    - CUDA_DEVICE_MAX_CONNECTIONS=1

    对 Pluto 代码的改动

    1. 已恢复（git checkout 回原始状态）：
    - src/models/pluto/layers/embedding.py — 之前删了 attn_drop 行，已恢复

    2. 仍然生效的改动：
    - src/models/pluto/modules/agent_encoder.py:83 — 第 83 行 torch.zeros 加了 dtype=x_agent_tmp.dtype，解决 FP16/BF16 下 index_put 的 dtype 不匹配问题
    - src/ 下所有子目录的 __init__.py — 手动创建的，原始仓库缺失这些文件
    - src/custom_training/__init__.py — 添加了 TrainingEngine 等 import