"""
简单快速验证 DecayingCosineAnnealingWarmRestarts 调度器
只需要基础的数值检查，不需要可视化
"""
import math
from toolkit.scheduler import DecayingCosineAnnealingWarmRestarts
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

def verify_decay_cosine_with_restart():
    """核心验证函数"""
    print("=" * 70)
    print("DecayingCosineAnnealingWarmRestarts 快速验证")
    print("=" * 70)
    
    # 测试配置
    initial_lr = 1e-4
    T_0 = 100
    T_mult = 2
    eta_min = 1e-7
    restart_decay = 0.8
    total_steps = 700
    
    print(f"\n配置参数:")
    print(f"  初始学习率 (initial_lr): {initial_lr:.2e}")
    print(f"  第一个周期长度 (T_0): {T_0}")
    print(f"  周期倍增因子 (T_mult): {T_mult}")
    print(f"  最小学习率 (eta_min): {eta_min:.2e}")
    print(f"  重启衰减因子 (restart_decay): {restart_decay}")
    print(f"  总步数: {total_steps}")
    
    # 创建调度器
    opt = create_optimizer(initial_lr)
    sch = DecayingCosineAnnealingWarmRestarts(
        opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min, restart_decay=restart_decay
    )
    
    # 收集学习率
    lrs = []
    for i in range(total_steps):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    # 检测 restart 点（学习率突然增加的点）
    jumps = []
    for i in range(1, len(lrs)):
        if lrs[i] - lrs[i-1] > 1e-7:
            jumps.append(i)
    
    # 计算预期的 restart 点
    expected_restarts = [T_0]
    period = T_0
    while expected_restarts[-1] + period * T_mult < total_steps:
        period *= T_mult
        expected_restarts.append(expected_restarts[-1] + period)
    
    print(f"\n✓ 检查 1: Restart 点位置")
    print(f"  预期 restart 在: {expected_restarts}")
    print(f"  实际 restart 在: {jumps}")
    if jumps == expected_restarts:
        print(f"  ✓ PASS - Restart 点完全正确!")
    else:
        print(f"  ⚠ WARN - Restart 点不完全匹配")
    
    # 检查每个 restart 后的学习率衰减
    print(f"\n✓ 检查 2: 每次 Restart 的学习率衰减")
    restart_lrs = [initial_lr]
    all_correct = True
    
    for idx, restart_step in enumerate(jumps):
        if restart_step < len(lrs):
            actual_lr = lrs[restart_step]
            expected_lr = initial_lr * (restart_decay ** (idx + 1))
            error = abs(actual_lr - expected_lr) / expected_lr
            
            status = "✓" if error < 0.001 else "✗"
            print(f"  Restart #{idx+1} (step {restart_step}): {actual_lr:.6e} (期望: {expected_lr:.6e}, 误差: {error:.2%}) {status}")
            
            if error >= 0.001:
                all_correct = False
            restart_lrs.append(actual_lr)
    
    if all_correct:
        print(f"  ✓ PASS - 所有 restart 的学习率衰减正确!")
    else:
        print(f"  ✗ FAIL - 部分 restart 的学习率衰减不正确")
    
    # 检查 cosine 衰减行为
    print(f"\n✓ 检查 3: Cosine 衰减行为")
    # 检查第一个周期内的衰减
    first_cycle_lrs = lrs[:T_0]
    print(f"  第一个周期 (0-{T_0}):")
    print(f"    起始 LR: {first_cycle_lrs[0]:.6e}")
    print(f"    中点 LR: {first_cycle_lrs[T_0//2]:.6e}")
    print(f"    结束 LR: {first_cycle_lrs[-1]:.6e}")
    
    # 验证 cosine 形状：起始应该最高，中点应该在中间，结束应该最低
    if first_cycle_lrs[0] > first_cycle_lrs[T_0//2] > first_cycle_lrs[-1]:
        print(f"  ✓ PASS - Cosine 衰减趋势正确")
    else:
        print(f"  ✗ FAIL - Cosine 衰减趋势不正确")
    
    # 检查最小学习率
    print(f"\n✓ 检查 4: 最小学习率约束")
    min_lr = min(lrs)
    print(f"  观察到的最小 LR: {min_lr:.6e}")
    print(f"  配置的 eta_min: {eta_min:.6e}")
    if min_lr >= eta_min * 0.99:  # 允许一点浮点误差
        print(f"  ✓ PASS - 学习率没有低于 eta_min")
    else:
        print(f"  ✗ FAIL - 学习率低于 eta_min")
    
    # 对比 PyTorch 原生调度器 (restart_decay=1.0)
    print(f"\n✓ 检查 5: 与 PyTorch 原生调度器对比 (restart_decay=1.0)")
    opt_pytorch = create_optimizer(initial_lr)
    sch_pytorch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_pytorch, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    
    opt_ours = create_optimizer(initial_lr)
    sch_ours = DecayingCosineAnnealingWarmRestarts(
        opt_ours, T_0=T_0, T_mult=T_mult, eta_min=eta_min, restart_decay=1.0
    )
    
    lrs_pytorch = []
    lrs_ours = []
    for i in range(300):
        sch_pytorch.step()
        sch_ours.step()
        lrs_pytorch.append(opt_pytorch.param_groups[0]['lr'])
        lrs_ours.append(opt_ours.param_groups[0]['lr'])
    
    max_diff = max(abs(a - b) for a, b in zip(lrs_pytorch, lrs_ours))
    print(f"  最大差异: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print(f"  ✓ PASS - restart_decay=1.0 时与 PyTorch 完全一致")
    else:
        print(f"  ✗ FAIL - 与 PyTorch 存在差异")
    
    # 总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    print("关键观察:")
    print(f"  • 检测到 {len(jumps)} 次 restart")
    print(f"  • 学习率范围: {min(lrs):.2e} ~ {max(lrs):.2e}")
    print(f"  • 第一次 restart 后的衰减: {restart_lrs[1]/restart_lrs[0]:.4f} (期望: {restart_decay})")
    
    if len(restart_lrs) > 2:
        print(f"  • 第二次 restart 后的衰减: {restart_lrs[2]/restart_lrs[1]:.4f} (期望: {restart_decay})")
    
    print("\n" + "=" * 70)
    
    # 返回用于进一步检查
    return {
        'lrs': lrs,
        'jumps': jumps,
        'restart_lrs': restart_lrs,
        'initial_lr': initial_lr,
        'restart_decay': restart_decay
    }

if __name__ == "__main__":
    result = verify_decay_cosine_with_restart()
    
    print("\n如果所有检查都通过，说明 DecayingCosineAnnealingWarmRestarts 工作正常！")
    print("\n提示：如果需要可视化，可以运行 test_scheduler_visual.py")
