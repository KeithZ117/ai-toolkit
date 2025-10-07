# 快速验证 DecayingCosineAnnealingWarmRestarts

## 🚀 在 Runpod 上验证

只需要一个命令：

```bash
python test_scheduler.py
```

## ✅ 期望输出

如果一切正常，你应该看到：

```
🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍
DecayingCosineAnnealingWarmRestarts 验证
🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍🔍

======================================================================
测试 1: 基础功能验证
======================================================================

配置: lr=1.00e-04, T_0=100, T_mult=2, restart_decay=0.8

检测到 3 个 restart 点: [100, 300, 700]

Restart 学习率验证:
  Restart 1 (step 100): 8.000000e-05 (期望: 8.000000e-05) ✓
  Restart 2 (step 300): 6.400000e-05 (期望: 6.400000e-05) ✓
  Restart 3 (step 700): 5.120000e-05 (期望: 5.120000e-05) ✓

学习率范围: 2.05e-07 ~ 1.00e-04

======================================================================
测试 2: PyTorch 兼容性 (restart_decay=1.0)
======================================================================

最大差异: 0.00e+00
✓ PASS - 与 PyTorch 完全一致

======================================================================
测试 3: 不同 restart_decay 参数
======================================================================

restart_decay=1.0:
  第1次 restart: 1.000000e-04
  第2次 restart: 1.000000e-04
  实际衰减: 1.0000 (期望: 1.0)
  ✓ PASS

restart_decay=0.8:
  第1次 restart: 8.000000e-05
  第2次 restart: 6.400000e-05
  实际衰减: 0.8000 (期望: 0.8)
  ✓ PASS

restart_decay=0.5:
  第1次 restart: 5.000000e-05
  第2次 restart: 2.500000e-05
  实际衰减: 0.5000 (期望: 0.5)
  ✓ PASS

======================================================================
测试 4: Cosine 曲线形状
======================================================================

第一个周期 (0-100):
  起始 LR: 1.000000e-04
  中点 LR: 5.000050e-05
  结束 LR: 2.000000e-07
  ✓ PASS - Cosine 曲线形状正确

======================================================================
测试总结
======================================================================
  基础功能: ✓ PASS
  PyTorch兼容性: ✓ PASS
  Restart衰减: ✓ PASS
  Cosine形状: ✓ PASS

总计: 4/4 测试通过

🎉 所有测试通过! DecayingCosineAnnealingWarmRestarts 工作正常!
```

## ❌ 如果看到错误

错误示例：
```
TypeError: DummyOpt is not an Optimizer
```

**解决方案：** 这个错误已经在最新的 `test_scheduler.py` 中修复了。确保你使用的是最新版本的测试脚本。

## 📝 关键指标

一个正常工作的实现应该显示：

1. **Restart 点正确** - 在 100, 300, 700 等位置
2. **衰减比例正确** - 每次 restart 后学习率 × restart_decay
3. **PyTorch 兼容** - restart_decay=1.0 时差异为 0
4. **Cosine 形状** - 学习率在周期内平滑下降

## 💡 使用提示

### 在训练代码中使用：

```python
from toolkit.scheduler import DecayingCosineAnnealingWarmRestarts

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 创建调度器 - 每次 restart 衰减到 80%
scheduler = DecayingCosineAnnealingWarmRestarts(
    optimizer,
    T_0=1000,           # 第一个周期 1000 步
    T_mult=2,           # 每次周期翻倍
    eta_min=1e-7,       # 最小学习率
    restart_decay=0.8   # 每次 restart 衰减到 80%
)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # ← 每个 batch 后调用
```

### 推荐参数：

| 训练长度 | T_0 | T_mult | restart_decay |
|---------|-----|--------|---------------|
| 短期 (<5000 步) | 100-500 | 1 | 0.9 |
| 中期 (5000-20000 步) | 1000 | 2 | 0.8 |
| 长期 (>20000 步) | 2000 | 2 | 0.7-0.8 |

## 🐛 故障排除

如果测试失败：

1. **Restart 点不对** → 检查 `step()` 函数中的周期计算
2. **衰减比例错误** → 检查 `_update_base_lrs()` 中的 factor 计算
3. **曲线不平滑** → 检查 `get_lr()` 中的 cosine 公式
4. **与 PyTorch 不一致** → 确保 restart_decay=1.0 时逻辑与原生一致

## 📊 可视化（可选）

如果需要看曲线图（需要 matplotlib）：

```bash
pip install matplotlib
python test_scheduler_visual.py
```

这会生成 `scheduler_validation.png` 文件。
