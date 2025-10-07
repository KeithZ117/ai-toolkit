import torch
from toolkit.scheduler import DecayingCosineAnnealingWarmRestarts

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p=torch.nn.Parameter(torch.tensor(0.0))

opt=torch.optim.SGD(M().parameters(), lr=1e-4)
sch=DecayingCosineAnnealingWarmRestarts(opt, T_0=100, T_mult=2, eta_min=1e-7, restart_decay=0.5)

lrs=[]
for i in range(710):
    sch.step()
    lrs.append(opt.param_groups[0]['lr'])

jumps=[i for i in range(1,len(lrs)) if lrs[i]-lrs[i-1] > 1e-7]
print('len', len(lrs))
print('last 5 lrs', lrs[-5:])
print('jumps', jumps)
print('lr at jumps', [lrs[i] for i in jumps])
