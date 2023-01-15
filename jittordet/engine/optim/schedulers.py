import inspect

import jittor.lr_scheduler as scheduler

from ..register import SCHEDULERS


def register_jittor_scheduler():
    for name, module in scheduler.__dict__.items():
        if inspect.isclass(module) and 'LR' in name:
            SCHEDULERS.register_module(name=name, module=module)


register_jittor_scheduler()


@SCHEDULERS.register_module()
class WarmUpLR:
    """Copy from JDet. Warm LR scheduler, which is the base lr_scheduler,
    default we use it.

    Args:
        optimizer (Optimizer): the optimizer to optimize the model
        warmup (string): Type of warmup used. It can be None(use no warmup),
            'constant', 'linear' or 'exp'
        warmup_iters (int): The number of iterations or epochs that warmup
            lasts
        warmup_ratio (float): LR used at the beginning of warmup equals to
            warmup_ratio * initial_lr
    """

    def __init__(self,
                 optimizer,
                 warmup_ratio=1.0 / 3,
                 warmup_iters=500,
                 warmup='linear',
                 last_iter=-1):
        self.optimizer = optimizer
        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.warmup = warmup
        self.base_lr = optimizer.lr
        self.base_lr_pg = [
            pg.get('lr', optimizer.lr) for pg in optimizer.param_groups
        ]
        self.by_iter = True
        self.last_iter = last_iter

    def get_warmup_lr(self, lr, cur_iters):
        if self.warmup == 'constant':
            k = self.warmup_ratio
        elif self.warmup == 'linear':
            k = 1 - (1 - cur_iters / self.warmup_iters) * (1 -
                                                           self.warmup_ratio)
        elif self.warmup == 'exp':
            k = self.warmup_ratio**(1 - cur_iters / self.warmup_iters)
        return k * lr

    def _update_lr(self, steps):
        self.optimizer.lr = self.get_warmup_lr(self.base_lr, steps)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.get_warmup_lr(self.base_lr_pg[i], steps)

    def step(self):
        self.last_iter += 1
        if self.last_iter <= self.warmup_iters:
            self._update_lr(self.last_iter)

    def state_dict(self):
        return {
            key: value
            for key, value in self.__dict__.items() if key != 'optimizer'
        }

    def load_state_dict(self, data):
        if isinstance(data, dict):
            for k, d in data.items():
                if k in self.__dict__:
                    self.__dict__[k] = d
