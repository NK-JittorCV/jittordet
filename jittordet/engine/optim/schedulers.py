import inspect

import jittor.lr_scheduler as scheduler

from ..register import SCHEDULERS


def register_jittor_scheduler():
    for name, module in scheduler.__dict__.items():
        if inspect.isclass(module) and 'LR' in name:
            SCHEDULERS.register_module(name=name, module=module)


register_jittor_scheduler()
