"""
快速验证 DecayingCosineAnnealingWarmRestarts 调度器
包含可视化和详细的数值检查
"""
import math
from toolkit.scheduler import DecayingCosineAnnealingWarmRestarts, get_lr_scheduler
import torch
import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单的模型用于优化器
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))

def create_optimizer(lr):
    """创建一个真实的优化器用于测试"""
    model = DummyModel()
    return torch.optim.SGD(model.parameters(), lr=lr)

def test_scheduler_basic():
    """基础功能测试"""
    print("=" * 60)
    print("测试 1: 基础功能验证")
    print("=" * 60)
    
    # 测试参数
    initial_lr = 1e-4
    T_0 = 100
    T_mult = 2
    eta_min = 1e-7
    restart_decay = 0.8
    total_steps = 700
    
    opt = create_optimizer(initial_lr)
    sch = DecayingCosineAnnealingWarmRestarts(
        opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min, restart_decay=restart_decay
    )
    
    lrs = []
    for i in range(total_steps):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    # 检查 restart 点
    jumps = [i for i in range(1, len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
    
    # 预期的 restart 点: T_0=100, 然后 200, 400...
    expected_restarts = [T_0]
    period = T_0
    while expected_restarts[-1] + period * T_mult < total_steps:
        period *= T_mult
        expected_restarts.append(expected_restarts[-1] + period)
    
    print(f"初始学习率: {initial_lr:.2e}")
    print(f"T_0: {T_0}, T_mult: {T_mult}")
    print(f"eta_min: {eta_min:.2e}, restart_decay: {restart_decay}")
    print(f"\n预期的 restart 点: {expected_restarts}")
    print(f"实际检测到的 restart 点: {jumps}")
    
    # 检查每个 restart 后学习率是否衰减
    print(f"\n每个 restart 的起始学习率:")
    restart_lrs = [initial_lr]
    for idx, restart_step in enumerate(jumps):
        if restart_step < len(lrs):
            lr_at_restart = lrs[restart_step]
            restart_lrs.append(lr_at_restart)
            expected_lr = initial_lr * (restart_decay ** (idx + 1))
            print(f"  Restart {idx+1} (step {restart_step}): {lr_at_restart:.2e} (期望: {expected_lr:.2e})")
    
    print(f"\n前 10 个学习率: {[f'{x:.2e}' for x in lrs[:10]]}")
    print(f"最后 10 个学习率: {[f'{x:.2e}' for x in lrs[-10:]]}")
    
    # 验证衰减比例
    if len(restart_lrs) > 1:
        actual_decay = restart_lrs[1] / restart_lrs[0]
        print(f"\n实际衰减比例: {actual_decay:.4f} (期望: {restart_decay})")
        if abs(actual_decay - restart_decay) < 0.01:
            print("✓ 衰减比例正确")
        else:
            print("✗ 衰减比例不正确")
    
    return lrs, jumps

def test_scheduler_edge_cases():
    """边界情况测试"""
    print("\n" + "=" * 60)
    print("测试 2: 边界情况")
    print("=" * 60)
    
    # 测试 restart_decay = 1.0 (不衰减)
    print("\n2.1: restart_decay=1.0 (不衰减)")
    opt = create_optimizer(1e-4)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1, eta_min=1e-7, restart_decay=1.0)
    lrs = []
    for i in range(200):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    jumps = [i for i in range(1, len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
    print(f"Restart 点: {jumps[:5]}")
    if len(jumps) >= 2:
        print(f"第一个 restart 的 LR: {lrs[jumps[0]]:.2e}")
        print(f"第二个 restart 的 LR: {lrs[jumps[1]]:.2e}")
        if abs(lrs[jumps[0]] - lrs[jumps[1]]) < 1e-9:
            print("✓ restart_decay=1.0 时学习率不衰减")
        else:
            print("✗ restart_decay=1.0 但学习率仍在衰减")
    
    # 测试 T_mult = 1 (固定周期)
    print("\n2.2: T_mult=1 (固定周期)")
    opt = create_optimizer(1e-4)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=50, T_mult=1, eta_min=1e-7, restart_decay=0.9)
    lrs = []
    for i in range(200):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    
    jumps = [i for i in range(1, len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
    print(f"Restart 点: {jumps}")
    periods = [jumps[i+1] - jumps[i] for i in range(len(jumps)-1)]
    if periods and all(p == 50 for p in periods):
        print("✓ T_mult=1 时周期保持固定")
    else:
        print(f"✗ T_mult=1 但周期不固定: {periods}")

def visualize_scheduler():
    """可视化学习率曲线"""
    print("\n" + "=" * 60)
    print("测试 3: 可视化")
    print("=" * 60)
    
    configs = [
        {'T_0': 100, 'T_mult': 2, 'restart_decay': 1.0, 'title': 'restart_decay=1.0 (无衰减)'},
        {'T_0': 100, 'T_mult': 2, 'restart_decay': 0.8, 'title': 'restart_decay=0.8'},
        {'T_0': 100, 'T_mult': 2, 'restart_decay': 0.5, 'title': 'restart_decay=0.5 (强衰减)'},
        {'T_0': 100, 'T_mult': 1, 'restart_decay': 0.8, 'title': 'T_mult=1 (固定周期)'},
        {
            'T_0': 100,
            'T_mult': 2,
            'restart_decay': 0.8,
            'warmup_steps': 50,
            'warmup_start_factor': 0.1,
            'title': 'warmup_steps=50 + restart_decay=0.8',
        },
    ]
    
    initial_lr = 1e-4
    eta_min = 1e-7
    total_steps = 700
    
    cols = 2
    rows = math.ceil(len(configs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle('DecayingCosineAnnealingWarmRestarts Scheduler 验证', fontsize=16)

    for idx, config in enumerate(configs):
        ax = axes[idx]

        opt = create_optimizer(initial_lr)
        if config.get('warmup_steps'):
            sch = get_lr_scheduler(
                "decaying_cosine_with_restarts",
                opt,
                T_0=config['T_0'],
                T_mult=config['T_mult'],
                eta_min=eta_min,
                restart_decay=config['restart_decay'],
                warmup_steps=config['warmup_steps'],
                warmup_start_factor=config.get('warmup_start_factor', 0.0),
            )
        else:
            sch = DecayingCosineAnnealingWarmRestarts(
                opt, T_0=config['T_0'], T_mult=config['T_mult'], 
                eta_min=eta_min, restart_decay=config['restart_decay']
            )

        lrs = []
        for i in range(total_steps):
            sch.step()
            lrs.append(opt.param_groups[0]['lr'])
        
        ax.plot(lrs, linewidth=2)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Learning Rate')
        ax.set_title(config['title'])
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        warmup_steps = config.get('warmup_steps', 0)
        if warmup_steps:
            ax.axvspan(0, warmup_steps, color='orange', alpha=0.12, label='warmup')

        jumps = []
        for i in range(1, len(lrs)):
            if warmup_steps and i <= warmup_steps:
                continue
            prev_lr = lrs[i - 1]
            curr_lr = lrs[i]
            if prev_lr <= 0:
                continue
            if curr_lr - prev_lr > 1e-6 and curr_lr / prev_lr > 1.5:
                jumps.append(i)
        for jump in jumps:
            ax.axvline(x=jump, color='r', linestyle='--', alpha=0.5, linewidth=1)

    for idx in range(len(configs), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图片
    output_path = 'scheduler_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化图表已保存到: {output_path}")
    
    # 尝试显示图表
    try:
        plt.show()
    except:
        print("(无法显示图表窗口，请查看保存的图片)")

def test_comparison_with_pytorch():
    """与 PyTorch 原生调度器对比"""
    print("\n" + "=" * 60)
    print("测试 4: 与 PyTorch CosineAnnealingWarmRestarts 对比")
    print("=" * 60)
    
    initial_lr = 1e-4
    T_0 = 100
    T_mult = 2
    eta_min = 1e-7
    total_steps = 300
    
    # PyTorch 原生调度器
    opt1 = create_optimizer(initial_lr)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt1, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    lrs_pytorch = []
    for i in range(total_steps):
        sch1.step()
        lrs_pytorch.append(opt1.param_groups[0]['lr'])
    
    # 我们的调度器 (restart_decay=1.0 时应该一样)
    opt2 = create_optimizer(initial_lr)
    sch2 = DecayingCosineAnnealingWarmRestarts(
        opt2, T_0=T_0, T_mult=T_mult, eta_min=eta_min, restart_decay=1.0
    )
    lrs_ours = []
    for i in range(total_steps):
        sch2.step()
        lrs_ours.append(opt2.param_groups[0]['lr'])
    
    # 计算差异
    diff = [abs(a - b) for a, b in zip(lrs_pytorch, lrs_ours)]
    max_diff = max(diff)
    avg_diff = sum(diff) / len(diff)
    
    print(f"最大差异: {max_diff:.2e}")
    print(f"平均差异: {avg_diff:.2e}")
    
    if max_diff < 1e-10:
        print("✓ restart_decay=1.0 时与 PyTorch 原生调度器完全一致")
    else:
        print("✗ 与 PyTorch 原生调度器存在差异")
        print(f"前 10 个差异: {[f'{x:.2e}' for x in diff[:10]]}")

def run_all_tests():
    """运行所有测试"""
    print("\n" + "🔍" * 30)
    print("DecayingCosineAnnealingWarmRestarts 完整验证")
    print("🔍" * 30 + "\n")
    
    # 运行测试
    lrs, jumps = test_scheduler_basic()
    test_scheduler_edge_cases()
    test_comparison_with_pytorch()
    
    # 可视化
    try:
        visualize_scheduler()
    except Exception as e:
        print(f"\n⚠ 可视化失败: {e}")
        print("可能需要安装 matplotlib: pip install matplotlib")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成!")
    print("=" * 60)
    
    return lrs, jumps

if __name__ == "__main__":
    lrs, jumps = run_all_tests()
