# DecayingCosineAnnealingWarmRestarts 快速验证指南

## 🎯 目的
快速验证你实现的 `DecayingCosineAnnealingWarmRestarts` 学习率调度器是否工作正常。

## 📋 最快验证方法

### ⚡ 推荐：一键验证（适用于 runpod 等所有环境）

```bash
python test_scheduler.py
```

**特点：**
- ✅ 30 秒内完成所有核心测试
- ✅ 4 个关键功能验证
- ✅ 清晰的 PASS/FAIL 输出
- ✅ 自动与 PyTorch 原生调度器对比
- ✅ 无需额外依赖（不需要 matplotlib）

**测试内容：**
1. 基础功能（Restart 点和衰减比例）
2. PyTorch 兼容性（restart_decay=1.0）
3. 不同 restart_decay 参数测试
4. Cosine 曲线形状验证

---

## 📋 其他验证方法

```bash
python test_scheduler_simple.py
```

**特点：**
- ✅ 详细的数值输出
- ✅ 显示每个 restart 点的学习率
- ✅ 检查衰减比例
- ✅ 与 PyTorch 原生调度器对比

**输出示例：**
```
✓ 检查 1: Restart 点位置
  预期 restart 在: [100, 300, 700]
  实际 restart 在: [100, 300, 700]
  ✓ PASS - Restart 点完全正确!

✓ 检查 2: 每次 Restart 的学习率衰减
  Restart #1 (step 100): 8.000000e-05 (期望: 8.000000e-05, 误差: 0.00%) ✓
  Restart #2 (step 300): 6.400000e-05 (期望: 6.400000e-05, 误差: 0.00%) ✓
```

---

### 方法 3: 可视化验证（需要 matplotlib）

```bash
# 先安装 matplotlib（如果还没有）
pip install matplotlib

# 运行可视化脚本
python test_scheduler_visual.py
```

**特点：**
- ✅ 生成学习率曲线图
- ✅ 对比不同 restart_decay 参数
- ✅ 可视化验证 cosine 形状和 restart 点
- ✅ 保存图片到 `scheduler_validation.png`

**生成 5 个对比图：**
1. restart_decay=1.0（无衰减）
2. restart_decay=0.8（中等衰减）
3. restart_decay=0.5（强衰减）
4. T_mult=1（固定周期）
5. warmup_steps=50 + restart_decay=0.8（带预热）

---

### 方法 4: 原始测试脚本

```bash
python tmp_sch.py
```

你原有的简单测试脚本，快速查看基本行为。

---

## 🔍 关键验证点

一个正常工作的 `DecayingCosineAnnealingWarmRestarts` 应该满足：

### 1. **Restart 点正确**
- 第一个 restart 在 step `T_0`
- 如果 `T_mult=2`，后续 restart 在 `T_0 + 2*T_0`, `T_0 + 2*T_0 + 4*T_0`, ...
- 如果 `T_mult=1`，每隔 `T_0` 就有一个 restart

### 2. **衰减正确**
- 每次 restart 后，最大学习率应该乘以 `restart_decay`
- 例如：初始 `1e-4`，`restart_decay=0.8`
  - 第 1 次 restart: `8e-5`
  - 第 2 次 restart: `6.4e-5`
  - 第 3 次 restart: `5.12e-5`

### 3. **Cosine 形状**
- 每个周期内，学习率应该按 cosine 曲线从高到低平滑下降
- 不应该有突然的跳跃（除了 restart 点）

### 4. **最小学习率约束**
- 学习率永远不应该低于 `eta_min`

### 5. **与 PyTorch 兼容**
- 当 `restart_decay=1.0` 时，应该与 `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` 完全一致

### 6. **Warmup（可选）**
- 如果设置了 `warmup_steps`，预热阶段的学习率应该线性增长到基础学习率
- 预热结束后再进入 cosine 退火周期，重启时依旧遵循 `restart_decay`

---

## 📊 快速诊断

如果测试失败，查看以下内容：

### ❌ Restart 点不对
→ 检查 `step()` 函数中的周期计算逻辑

### ❌ 衰减比例不对
→ 检查 `_update_base_lrs()` 函数中的 `factor` 计算

### ❌ 学习率曲线不平滑
→ 检查 `get_lr()` 函数中的 cosine 公式

### ❌ 最小学习率被突破
→ 检查 `eta_min` 是否也在衰减（应该衰减）

---

## 💡 使用示例

在训练代码中使用：

```python
from toolkit.scheduler import (
    DecayingCosineAnnealingWarmRestarts,
    get_lr_scheduler,
)

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 方案 A：纯 decaying cosine with restarts
scheduler = DecayingCosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,           # 第一个周期 1000 步
    T_mult=2,           # 每次周期翻倍
    eta_min=1e-7,       # 最小学习率
    restart_decay=0.8   # 每次 restart 衰减到 80%
)

# 方案 B：加入 warmup，只在开头预热一次
# scheduler = get_lr_scheduler(
#     "decaying_cosine_with_restarts",
#     optimizer,
#     T_0=1000,
#     T_mult=2,
#     eta_min=1e-7,
#     restart_decay=0.8,
#     warmup_steps=500,
#     warmup_start_factor=0.1,
# )

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # ... 训练代码 ...
        optimizer.step()
        scheduler.step()  # 每个 batch 后调用
```

---

## 🎉 期望结果

如果所有测试都通过，你应该看到：

```
🎉 所有测试通过! DecayingCosineAnnealingWarmRestarts 工作正常!
```

这表示你的实现：
- ✅ Restart 机制正常
- ✅ 衰减计算正确
- ✅ Cosine 曲线形状正确
- ✅ 与 PyTorch 兼容
- ✅ 可以安全使用

---

## 📝 参数说明

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `T_0` | 第一个 restart 周期的长度 | 100-1000 |
| `T_mult` | 周期倍增因子 | 1 或 2 |
| `eta_min` | 最小学习率 | 1e-7 |
| `restart_decay` | Restart 衰减因子 | 0.5-1.0 |
| `warmup_steps` | 预热步数（可选） | 0-1000 |
| `warmup_start_factor` | 预热起始系数（可选） | 0.0-0.1 |

**建议：**
- 对于长训练：`T_0=1000`, `T_mult=2`, `restart_decay=0.8`
- 对于短训练：`T_0=100`, `T_mult=1`, `restart_decay=0.9`
- 不想衰减：`restart_decay=1.0`（等同于 PyTorch 原生）
- 需要更平滑的起步：加入 `warmup_steps`（例如 500）和 `warmup_start_factor=0.1`

---

## 🐛 调试技巧

如果想看学习率变化：

```python
scheduler = DecayingCosineAnnealingWarmRestarts(...)
lrs = []
for i in range(500):
    scheduler.step()
    lrs.append(optimizer.param_groups[0]['lr'])

# 绘制曲线
import matplotlib.pyplot as plt
plt.plot(lrs)
plt.yscale('log')
plt.show()
```
