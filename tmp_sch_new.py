import math
from toolkit.scheduler import DecayingCosineAnnealingWarmRestarts
import torch

class DummyModel(torch.nn.Module):
    def __init__(self):`n        super().__init__()`n        self.param = torch.nn.Parameter(torch.tensor(0.0))`n`ndef create_optimizer(lr):`n    """创建一个真实的优化器用于测试"""`n    model = DummyModel()`n    return torch.optim.SGD(model.parameters(), lr=lr)`n`nclass _OLD:    def __init__(self, lr):
        self.param_groups = [{"lr": lr, "initial_lr": lr}]

for T0 in (420, 6300):
    print('--- T0', T0)
    opt = DummyOpt(7.5e-5)
    sch = DecayingCosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=2, eta_min=2e-7, restart_decay=0.8)
    lrs = []
    for i in range(6300):
        sch.step()
        lrs.append(opt.param_groups[0]['lr'])
    jumps = [i for i in range(1,len(lrs)) if lrs[i] - lrs[i-1] > 1e-7]
    print('jumps at:', jumps[:10])
    print('first 5 lrs:', [f"{x:.2e}" for x in lrs[:5]])
    print('last 5 lrs:', [f"{x:.2e}" for x in lrs[-5:]])
