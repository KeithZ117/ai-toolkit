"""
DecayingCosineAnnealingWarmRestarts 快速验证脚本
适用于 runpod 或任何环境
"""
import math
from toolkit.scheduler import DecayingCosineAnnealingWarmRestarts, get_lr_scheduler
import torch

# 创建简单的模型和优化器
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))

def create_optimizer(lr):
    model = SimpleModel()
    return torch.optim.SGD(model.parameters(), lr=lr)

def test_basic():
    """基础功能测试"""
    print("=" * 70)
    print("测试 1: 基础功能验证")
    print("=" * 70)
    
    initial_lr = 1e-4
    T_0 = 100
    T_mult = 2
    eta_min = 1e-7
    restart_decay = 0.8
    total_steps = 700
    
    print(f"\n配置: lr={initial_lr:.2e}, T_0={T_0}, T_mult={T_mult}, restart_decay={restart_decay}")
    
    opt = create_optimizer(initial_lr)
    sch = DecayingCosineAnnealingWarmRestarts(
        opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min, restart_decay=restart_decay
    )
    
    lrs = []
    for i in range(total_steps):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    # 检测 restart 点
    jumps = [i for i in range(1, len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
    
    print(f"\n检测到 {len(jumps)} 个 restart 点: {jumps}")
    
    # 检查每个 restart 的学习率
    restart_lrs = [initial_lr]
    print(f"\nRestart 学习率验证:")
    for idx, restart_step in enumerate(jumps):
        if restart_step < len(lrs):
            actual_lr = lrs[restart_step]
            expected_lr = initial_lr * (restart_decay ** (idx + 1))
            error = abs(actual_lr - expected_lr) / expected_lr
            status = "✓" if error < 0.01 else "✗"
            print(f"  Restart {idx+1} (step {restart_step}): {actual_lr:.6e} (期望: {expected_lr:.6e}) {status}")
            restart_lrs.append(actual_lr)
    
    print(f"\n学习率范围: {min(lrs):.2e} ~ {max(lrs):.2e}")
    
    return len(jumps) >= 2 and all(abs(restart_lrs[i+1]/restart_lrs[i] - restart_decay) < 0.01 for i in range(len(restart_lrs)-1))

def test_pytorch_compatibility():
    """与 PyTorch 原生调度器对比"""
    print("\n" + "=" * 70)
    print("测试 2: PyTorch 兼容性 (restart_decay=1.0)")
    print("=" * 70)
    
    initial_lr = 1e-4
    T_0 = 100
    T_mult = 2
    eta_min = 1e-7
    
    # PyTorch 原生
    opt1 = create_optimizer(initial_lr)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt1, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    
    # 我们的实现 (restart_decay=1.0)
    opt2 = create_optimizer(initial_lr)
    sch2 = DecayingCosineAnnealingWarmRestarts(
        opt2, T_0=T_0, T_mult=T_mult, eta_min=eta_min, restart_decay=1.0
    )
    
    max_diff = 0
    for i in range(300):
        sch1.step()
        sch2.step()
        diff = abs(opt1.param_groups[0]['lr'] - opt2.param_groups[0]['lr'])
        max_diff = max(max_diff, diff)
    
    print(f"\n最大差异: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✓ PASS - 与 PyTorch 完全一致")
        return True
    else:
        print("✗ FAIL - 与 PyTorch 存在差异")
        return False

def test_restart_decay():
    """测试不同 restart_decay 值"""
    print("\n" + "=" * 70)
    print("测试 3: 不同 restart_decay 参数")
    print("=" * 70)
    
    initial_lr = 1e-4
    T_0 = 50
    
    for decay in [1.0, 0.8, 0.5]:
        print(f"\nrestart_decay={decay}:")
        opt = create_optimizer(initial_lr)
        sch = DecayingCosineAnnealingWarmRestarts(
            opt, T_0=T_0, T_mult=1, eta_min=1e-7, restart_decay=decay
        )
        
        lrs = []
        for i in range(250):
            sch.step()
            lrs.append(opt.param_groups[0]['lr'])
        
        jumps = [i for i in range(1, len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
        
        if len(jumps) >= 2:
            first_restart_lr = lrs[jumps[0]]
            second_restart_lr = lrs[jumps[1]]
            actual_decay = second_restart_lr / first_restart_lr
            print(f"  第1次 restart: {first_restart_lr:.6e}")
            print(f"  第2次 restart: {second_restart_lr:.6e}")
            print(f"  实际衰减: {actual_decay:.4f} (期望: {decay})")
            if abs(actual_decay - decay) < 0.01:
                print(f"  ✓ PASS")
            else:
                print(f"  ✗ FAIL")

def test_warmup_with_decaying():
    """验证 warmup + decaying cosine with restarts 的组合"""
    print("\n" + "=" * 70)
    print("测试 4: Warmup + Decaying Cosine")
    print("=" * 70)

    initial_lr = 1e-4
    T_0 = 50
    eta_min = 1e-7
    restart_decay = 0.8
    warmup_steps = 10
    warmup_start_factor = 0.1

    opt = create_optimizer(initial_lr)
    sch = get_lr_scheduler(
        "decaying_cosine_with_restarts",
        opt,
        T_0=T_0,
        T_mult=1,
        eta_min=eta_min,
        restart_decay=restart_decay,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
    )

    total_steps = warmup_steps + 2 * T_0
    lrs = []
    for _ in range(total_steps):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])

    warmup_phase = lrs[:warmup_steps]
    cosine_phase = lrs[warmup_steps:]

    print(f"\nWarmup 阶段 (0-{warmup_steps}):")
    print(f"  起始 LR: {warmup_phase[0]:.6e}")
    print(f"  结束 LR: {warmup_phase[-1]:.6e}")

    is_warmup_increasing = all(
        warmup_phase[i] <= warmup_phase[i + 1] + 1e-12
        for i in range(len(warmup_phase) - 1)
    )
    warmup_reaches_target = abs(warmup_phase[-1] - initial_lr) < initial_lr * 0.01

    print(f"  单调递增: {'✓' if is_warmup_increasing else '✗'}")
    print(f"  收敛至初始 LR: {'✓' if warmup_reaches_target else '✗'}")

    first_cycle = cosine_phase[:T_0]
    second_cycle = cosine_phase[T_0:]

    print(f"\nCosine 第一周期样本:")
    print(f"  起点: {first_cycle[0]:.6e}")
    print(f"  中点: {first_cycle[T_0 // 2]:.6e}")
    print(f"  终点: {first_cycle[-1]:.6e}")

    cosine_decreasing = all(first_cycle[i] >= first_cycle[i + 1] for i in range(len(first_cycle) - 1))
    print(f"  单调递减: {'✓' if cosine_decreasing else '✗'}")

    # 检测 restart 位置 (忽略 warmup 小幅上升)
    jump_threshold = initial_lr * 0.2
    jumps = [
        idx for idx in range(1, len(lrs))
        if lrs[idx] - lrs[idx - 1] > jump_threshold
    ]

    expected_first_restart = warmup_steps + T_0 - 1
    expected_second_restart = expected_first_restart + T_0

    print(f"\n检测到的 restart 点: {jumps}")
    print(f"期望 restart 点: [{expected_first_restart}, {expected_second_restart}]")

    restart_ok = (
        len(jumps) >= 2
        and jumps[0] == expected_first_restart
        and jumps[1] == expected_second_restart
    )

    print(f"  Restart 位置正确: {'✓' if restart_ok else '✗'}")

    # 复核 restart 学习率是否按 restart_decay 衰减
    if restart_ok:
        first_restart_lr = lrs[jumps[0]]
        second_restart_lr = lrs[jumps[1]]
        ratio = second_restart_lr / first_restart_lr
        print(f"  Restart 学习率: {first_restart_lr:.6e} -> {second_restart_lr:.6e}")
        print(f"  实际衰减: {ratio:.4f} (期望: {restart_decay})")
        decay_ok = abs(ratio - restart_decay) < 0.01
    else:
        decay_ok = False

    all_pass = is_warmup_increasing and warmup_reaches_target and cosine_decreasing and restart_ok and decay_ok

    if all_pass:
        print("\n✓ PASS - Warmup + Decaying Cosine 行为正确")
        return True
    else:
        print("\n✗ FAIL - Warmup + Decaying Cosine 行为存在问题")
        return False

def test_cosine_shape():
    """测试 cosine 曲线形状"""
    print("\n" + "=" * 70)
    print("测试 5: Cosine 曲线形状")
    print("=" * 70)
    
    T_0 = 100
    opt = create_optimizer(1e-4)
    sch = DecayingCosineAnnealingWarmRestarts(
        opt, T_0=T_0, T_mult=1, eta_min=1e-7, restart_decay=0.8
    )
    
    lrs = []
    for i in range(T_0):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    print(f"\n第一个周期 (0-{T_0}):")
    print(f"  起始 LR: {lrs[0]:.6e}")
    print(f"  中点 LR: {lrs[T_0//2]:.6e}")
    print(f"  结束 LR: {lrs[-1]:.6e}")
    
    # Cosine 曲线应该单调递减
    is_decreasing = all(lrs[i] >= lrs[i+1] for i in range(len(lrs)-1))
    
    if is_decreasing and lrs[0] > lrs[T_0//2] > lrs[-1]:
        print("  ✓ PASS - Cosine 曲线形状正确")
        return True
    else:
        print("  ✗ FAIL - Cosine 曲线形状不正确")
        return False

def main():
    """运行所有测试"""
    print("\n" + "🔍" * 35)
    print("DecayingCosineAnnealingWarmRestarts 验证")
    print("🔍" * 35 + "\n")
    
    results = []
    
    try:
        results.append(("基础功能", test_basic()))
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        results.append(("基础功能", False))
    
    try:
        results.append(("PyTorch兼容性", test_pytorch_compatibility()))
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        results.append(("PyTorch兼容性", False))
    
    try:
        test_restart_decay()
        results.append(("Restart衰减", True))
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        results.append(("Restart衰减", False))

    try:
        results.append(("Warmup组合", test_warmup_with_decaying()))
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        results.append(("Warmup组合", False))

    try:
        results.append(("Cosine形状", test_cosine_shape()))
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        results.append(("Cosine形状", False))
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! DecayingCosineAnnealingWarmRestarts 工作正常!")
        return 0
    else:
        print(f"\n⚠ 有 {total - passed} 个测试失败")
        return 1

if __name__ == "__main__":
    exit(main())
