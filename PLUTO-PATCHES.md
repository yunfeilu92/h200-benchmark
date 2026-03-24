# Pluto 代码修改补丁

在 g6e (L40S) 上运行 Pluto benchmark 需要的所有代码修改。
基于 [pluto](https://github.com/jchengai/pluto) 原始代码。

---

## 1. FPN 上采样：linear → nearest（关键性能优化）

**文件**: `src/models/pluto/layers/embedding.py`
**行**: 78-83
**原因**: `F.interpolate(mode="linear")` 占 63.6% GPU 时间，是压倒性瓶颈。改 nearest 后训练加速 47%，val_loss 更低。

```diff
         for i in range(len(out) - 1, 0, -1):
             laterals[i - 1] = laterals[i - 1] + F.interpolate(
                 laterals[i],
                 scale_factor=(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
-                mode="linear",
-                align_corners=False,
+                mode="nearest",
             )
```

---

## 2. custom_training/__init__.py 添加 import

**文件**: `src/custom_training/__init__.py`
**原因**: 原始文件为空，`run_training.py` 需要从该模块导入 `TrainingEngine` 等。

```python
# 原始: 空文件

# 修改为:
from src.custom_training.custom_training_builder import (
    TrainingEngine,
    build_training_engine,
    update_config_for_training,
)
```

---

## 3. 创建缺失的 __init__.py 文件

**目录**: `src/` 下所有子目录
**原因**: 原始仓库缺少部分 `__init__.py`，导致 import 失败。

```bash
find src -type d -exec sh -c 'touch "$1/__init__.py" 2>/dev/null' _ {} \;
```

---

## 4. BF16 dtype 兼容性修复（5 处）

> 注意：FP32 训练下非必须。仅在使用 `precision=bf16-mixed` 时需要。

### 4a. agent_encoder.py

**文件**: `src/models/pluto/modules/agent_encoder.py`
**行**: 83

```diff
-        x_agent = torch.zeros(bs * A, self.dim, device=position.device)
+        x_agent = torch.zeros(bs * A, self.dim, device=position.device, dtype=x_agent_tmp.dtype)
```

### 4b. embedding.py (两处)

**文件**: `src/models/pluto/layers/embedding.py`
**行**: 283

```diff
-        x_features = torch.zeros(bs, n, 256, device=device)
+        x_features = torch.zeros(bs, n, 256, device=device, dtype=x_valid.dtype)
```

**行**: 292

```diff
-        res = torch.zeros(bs, n, self.encoder_channel, device=device)
+        res = torch.zeros(bs, n, self.encoder_channel, device=device, dtype=x_features_valid.dtype)
```

### 4c. map_encoder.py (两处)

**文件**: `src/models/pluto/modules/map_encoder.py`
**行**: 85

```diff
-        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device)
+        x_speed_limit = torch.zeros(bs, M, self.dim, device=x_polygon.device, dtype=x_polygon.dtype)
```

**行**: 89

```diff
-        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight
+        x_speed_limit[~polygon_has_speed_limit] = self.unknown_speed_emb.weight.to(x_speed_limit.dtype)
```

---

## 5. ConvTranspose1d 替换方案（可选，替代 nearest）

> 和 patch #1 互斥。ConvTranspose1d 速度与 nearest 相同，但 val_loss 略高（3.783 vs 3.617）。
> 推荐使用 patch #1 (nearest)。

### 5a. embedding.py __init__ 添加 upsample_convs

**文件**: `src/models/pluto/layers/embedding.py`
**位置**: `self.fpn_conv = nn.Conv1d(n, n, 3, padding=1)` 之后

```python
        # Learnable upsample: ConvTranspose1d 替代 F.interpolate(mode="linear")
        self.upsample_convs = nn.ModuleList()
        for i in range(len(out_indices) - 1):
            self.upsample_convs.append(
                nn.ConvTranspose1d(n, n, kernel_size=3, stride=2, padding=1)
            )
```

### 5b. embedding.py forward 替换 interpolate 循环

```diff
         for i in range(len(out) - 1, 0, -1):
-            laterals[i - 1] = laterals[i - 1] + F.interpolate(
-                laterals[i],
-                scale_factor=(laterals[i - 1].shape[-1] / laterals[i].shape[-1]),
-                mode="linear",
-                align_corners=False,
-            )
+            target_size = laterals[i - 1].shape[-1]
+            upsampled = self.upsample_convs[i - 1](
+                laterals[i], output_size=(target_size,)
+            )
+            laterals[i - 1] = laterals[i - 1] + upsampled
```

### 5c. pluto_trainer.py 添加 ConvTranspose1d 到 whitelist

**文件**: `src/models/pluto/pluto_trainer.py`
**位置**: `whitelist_weight_modules` 元组中

```diff
         whitelist_weight_modules = (
             nn.Linear,
             nn.Conv1d,
+            nn.ConvTranspose1d,
             nn.Conv2d,
             nn.Conv3d,
             nn.MultiheadAttention,
```

---

## 6. 环境配置（非代码修改）

### 6a. pluto.pth — Python path

```bash
# 让 Ray workers 能 import src.models
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
echo "/opt/dlami/nvme/pluto" > "$SITE/pluto.pth"
```

### 6b. rich console.py — safe pop

```bash
# 修复非终端模式下 RichProgressBar 崩溃
RICH_CONSOLE=$(python -c "import rich.console; print(rich.console.__file__)")
sed -i 's/self\._live_stack\.pop()/self._live_stack.pop() if self._live_stack else None/g' "$RICH_CONSOLE"
```

### 6c. 数据目录符号链接

```bash
# nuplan 期望 /nuplan/dataset/nuplan-v1.1/trainval/
mkdir -p /nuplan/dataset/nuplan-v1.1/trainval
for dir in /nuplan/dataset/data/cache/train_* /nuplan/dataset/data/cache/val; do
    for db in "$dir"/*.db; do
        ln -sf "$db" /nuplan/dataset/nuplan-v1.1/trainval/
    done
done
```

---

## 应用顺序

最小必要修改（FP32 + nearest 最优配置）：
1. **Patch #1** — nearest（必须，性能关键）
2. **Patch #2** — custom_training __init__（必须，否则无法启动）
3. **Patch #3** — 创建 __init__.py（必须）
4. **Patch 6a** — pluto.pth（必须，Ray workers 需要）
5. **Patch 6b** — rich safe pop（推荐，否则非终端下可能崩溃）
6. **Patch 6c** — 数据目录链接（必须，否则找不到数据）

BF16 额外需要：
7. **Patch #4** — 所有 dtype 修复（5 处）

ConvTranspose1d 方案（替代 #1）：
8. **Patch #5** — ConvTranspose1d（替代 nearest，不推荐）
