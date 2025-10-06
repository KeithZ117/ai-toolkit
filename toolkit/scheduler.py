import math
import torch
from typing import Optional
from diffusers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION, get_constant_schedule_with_warmup
from torch.optim.lr_scheduler import (
    _LRScheduler,
    _enable_get_lr_call,
    _warn_get_lr_called_within_step,
)


class DecayingCosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        restart_decay: float = 1.0,
        last_epoch: int = -1,
    ) -> None:
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}"
            )
        if restart_decay <= 0:
            raise ValueError(
                f"Expected positive restart_decay, but got {restart_decay}"
            )

        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self._initial_eta_min = eta_min  # Store initial eta_min
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.restart_decay = restart_decay
        self._cycle = 0
        self._initial_base_lrs = None
        super().__init__(optimizer, last_epoch)

        if self._initial_base_lrs is None:
            self._initial_base_lrs = list(self.base_lrs)

    def _update_base_lrs(self) -> None:
        if self._initial_base_lrs is None:
            self._initial_base_lrs = list(self.base_lrs)
        factor = self.restart_decay ** self._cycle
        self.base_lrs = [base * factor for base in self._initial_base_lrs]
        # Decay eta_min with the same factor
        self.eta_min = self._initial_eta_min * factor
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["initial_lr"] = base

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur % self.T_i
                self.T_i = self.T_i * self.T_mult
                self._cycle += 1
                self._update_base_lrs()
        else:
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")

            prev_cycle = self._cycle
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self._cycle = int(epoch // self.T_0)
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(
                        math.log(
                            epoch / self.T_0 * (self.T_mult - 1) + 1, self.T_mult
                        )
                    )
                    self._cycle = n
                    self.T_i = self.T_0 * self.T_mult ** n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
            else:
                self._cycle = 0
                self.T_i = self.T_0
                self.T_cur = epoch

            if self._cycle != prev_cycle:
                self._update_base_lrs()

        self.last_epoch = math.floor(epoch)

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


def get_lr_scheduler(
        name: Optional[str],
        optimizer: torch.optim.Optimizer,
        **kwargs,
):
    if name == "cosine":
        if 'total_iters' in kwargs:
            kwargs['T_max'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **kwargs
        )
    elif name == "cosine_with_restarts":
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name == "decaying_cosine_with_restarts":
        if 'total_iters' in kwargs:
            kwargs['T_0'] = kwargs.pop('total_iters')
        return DecayingCosineAnnealingWarmRestarts(
            optimizer, **kwargs
        )
    elif name == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer, **kwargs
        )
    elif name == "constant":
        if 'factor' not in kwargs:
            kwargs['factor'] = 1.0

        return torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    elif name == "linear":

        return torch.optim.lr_scheduler.LinearLR(
            optimizer, **kwargs
        )
    elif name == 'constant_with_warmup':
        # see if num_warmup_steps is in kwargs
        if 'num_warmup_steps' not in kwargs:
            print(f"WARNING: num_warmup_steps not in kwargs. Using default value of 1000")
            kwargs['num_warmup_steps'] = 1000
        del kwargs['total_iters']
        return get_constant_schedule_with_warmup(optimizer, **kwargs)
    else:
        # try to use a diffusers scheduler
        print(f"Trying to use diffusers scheduler {name}")
        try:
            name = SchedulerType(name)
            schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **kwargs)
        except Exception as e:
            print(e)
            pass
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, decaying_cosine_with_restarts, step, linear or constant"
        )
