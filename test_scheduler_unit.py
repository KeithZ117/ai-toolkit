"""
单元测试: DecayingCosineAnnealingWarmRestarts
快速验证核心功能是否正确
"""
import sys
import math
from toolkit.scheduler import DecayingCosineAnnealingWarmRestarts, get_lr_scheduler
import torch

# 创建一个简单的模型用于优化器
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))

def create_optimizer(lr):
    """创建一个真实的优化器用于测试"""
    model = DummyModel()
    return torch.optim.SGD(model.parameters(), lr=lr)

def test_basic_initialization():
    """测试基础初始化"""
    print("测试 1: 基础初始化...", end=" ")
    opt = create_optimizer(1e-4)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-7, restart_decay=0.8)
    assert sch.T_0 == 100
    assert sch.T_mult == 2
    assert sch.eta_min == 1e-7
    assert sch.restart_decay == 0.8
    print("✓ PASS")
    return True

def test_restart_decay():
    """测试 restart 后学习率是否正确衰减"""
    print("测试 2: Restart 衰减...", end=" ")
    
    initial_lr = 1e-4
    restart_decay = 0.8
    T_0 = 100
    
    opt = create_optimizer(initial_lr)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=2, eta_min=1e-7, restart_decay=restart_decay)
    
    # 运行到第一个 restart
    for _ in range(T_0):
        sch.step()
    
    # 下一步应该是 restart，学习率应该衰减
    sch.step()
    lr_after_first_restart = opt.param_groups[0]['lr']
    expected_lr = initial_lr * restart_decay
    
    # 允许 1% 的浮点误差
    error = abs(lr_after_first_restart - expected_lr) / expected_lr
    assert error < 0.01, f"学习率不匹配: {lr_after_first_restart:.6e} vs {expected_lr:.6e}"
    
    print("✓ PASS")
    return True

def test_period_multiplication():
    """测试周期倍增"""
    print("测试 3: 周期倍增...", end=" ")
    
    T_0 = 50
    T_mult = 2
    
    opt = create_optimizer(1e-4)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=1e-7, restart_decay=0.8)
    
    lrs = []
    for _ in range(250):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    # 检测 restart 点
    jumps = [i for i in range(1, len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
    
    # 预期: 50, 150 (50 + 100), ...
    expected_jumps = [50, 150]
    
    assert jumps[:2] == expected_jumps, f"Restart 点不正确: {jumps[:2]} vs {expected_jumps}"
    
    print("✓ PASS")
    return True

def test_eta_min_constraint():
    """测试最小学习率约束"""
    print("测试 4: 最小学习率约束...", end=" ")
    
    eta_min = 1e-6
    opt = create_optimizer(1e-4)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=eta_min, restart_decay=0.5)
    
    lrs = []
    for _ in range(1000):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    min_lr = min(lrs)
    
    # 最小学习率应该不低于 eta_min (允许微小的浮点误差)
    assert min_lr >= eta_min * 0.99, f"学习率低于 eta_min: {min_lr:.6e} < {eta_min:.6e}"
    
    print("✓ PASS")
    return True

def test_no_decay_compatibility():
    """测试 restart_decay=1.0 时与 PyTorch 原生调度器一致"""
    print("测试 5: PyTorch 兼容性 (restart_decay=1.0)...", end=" ")
    
    initial_lr = 1e-4
    T_0 = 100
    T_mult = 2
    eta_min = 1e-7
    
    # PyTorch 原生
    opt1 = create_optimizer(initial_lr)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt1, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    
    # 我们的实现
    opt2 = create_optimizer(initial_lr)
    sch2 = DecayingCosineAnnealingWarmRestarts(
        opt2, T_0=T_0, T_mult=T_mult, eta_min=eta_min, restart_decay=1.0
    )
    
    for _ in range(300):
        sch1.step()
        sch2.step()
        
        diff = abs(opt1.param_groups[0]['lr'] - opt2.param_groups[0]['lr'])
        assert diff < 1e-10, f"与 PyTorch 差异过大: {diff:.2e}"
    
    print("✓ PASS")
    return True

def test_cosine_shape():
    """测试 cosine 曲线形状"""
    print("测试 6: Cosine 曲线形状...", end=" ")
    
    T_0 = 100
    opt = create_optimizer(1e-4)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=1, eta_min=1e-7, restart_decay=0.8)
    
    lrs = []
    for _ in range(T_0):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    # Cosine 曲线应该：起始高，中间下降，结束最低
    assert lrs[0] > lrs[T_0//2], "Cosine 曲线形状不正确"
    assert lrs[T_0//2] > lrs[-1], "Cosine 曲线形状不正确"
    
    # 前半部分应该单调递减
    for i in range(1, T_0//2):
        assert lrs[i] <= lrs[i-1], f"前半部分应该单调递减，但在 step {i} 处增加了"
    
    print("✓ PASS")
    return True

def test_multiple_restarts():
    """测试多次 restart 的衰减"""
    print("测试 7: 多次 Restart 衰减...", end=" ")
    
    initial_lr = 1e-4
    restart_decay = 0.75
    T_0 = 50
    
    opt = create_optimizer(initial_lr)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=1, eta_min=1e-7, restart_decay=restart_decay)
    
    lrs = []
    for _ in range(250):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    # 检测 restart 点
    jumps = [i for i in range(1, len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
    
    # 至少应该有 3 次 restart
    assert len(jumps) >= 3, f"应该至少有 3 次 restart，但只有 {len(jumps)} 次"
    
    # 检查每次 restart 的衰减比例
    restart_lrs = [initial_lr] + [lrs[j] for j in jumps]
    
    for i in range(1, len(restart_lrs)):
        actual_ratio = restart_lrs[i] / restart_lrs[i-1]
        expected_ratio = restart_decay
        error = abs(actual_ratio - expected_ratio) / expected_ratio
        assert error < 0.01, f"第 {i} 次 restart 衰减比例不正确: {actual_ratio:.4f} vs {expected_ratio:.4f}"
    
    print("✓ PASS")
    return True

def test_warmup_integration():
    """测试 warmup + decaying cosine 组合"""
    print("测试 8: Warmup 组合...", end=" ")

    initial_lr = 1e-4
    warmup_steps = 8
    warmup_start_factor = 0.1

    opt = create_optimizer(initial_lr)
    sch = get_lr_scheduler(
        "decaying_cosine_with_restarts",
        opt,
        T_0=40,
        T_mult=1,
        eta_min=1e-7,
        restart_decay=0.8,
        warmup_steps=warmup_steps,
        warmup_start_factor=warmup_start_factor,
    )

    warmup_lrs = []
    for _ in range(warmup_steps):
        sch.step()
        warmup_lrs.append(opt.param_groups[0]['lr'])

    assert warmup_lrs[0] < warmup_lrs[-1] <= initial_lr + 1e-12
    assert all(warmup_lrs[i] <= warmup_lrs[i + 1] + 1e-12 for i in range(len(warmup_lrs) - 1))

    sch.step()
    lr_after_warmup = opt.param_groups[0]['lr']
    assert abs(lr_after_warmup - initial_lr) < initial_lr * 0.01

    print("✓ PASS")
    return True

def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("DecayingCosineAnnealingWarmRestarts 单元测试")
    print("=" * 60 + "\n")
    
    tests = [
        test_basic_initialization,
        test_restart_decay,
        test_period_multiplication,
        test_eta_min_constraint,
        test_no_decay_compatibility,
        test_cosine_shape,
        test_multiple_restarts,
        test_warmup_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"✗ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 所有测试通过! DecayingCosineAnnealingWarmRestarts 工作正常!")
        return 0
    else:
        print(f"\n⚠ 有 {failed} 个测试失败，需要检查实现。")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
