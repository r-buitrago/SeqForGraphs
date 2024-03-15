import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, base_scheduler):
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.last_step = -1
        super(WarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            # Warmup phase: linearly scale up the learning rate
            return [base_lr * (self.last_step + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # After warmup: use the base scheduler's learning rate
            return self.base_scheduler.get_last_lr()

    def step(self):
        self.last_step += 1
        if self.last_step >= self.warmup_steps:
            # Update base scheduler
            self.base_scheduler.step()
        super(WarmupScheduler, self).step()
