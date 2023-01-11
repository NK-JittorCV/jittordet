import inspect

import jittor.optim as optim

from ..register import OPTIMIZERS


def register_jittor_optim():
    for name, module in optim.__dict__.items():
        if not inspect.isclass(module):
            continue

        if issubclass(module,
                      optim.Optimizer) and module is not optim.Optimizer:
            OPTIMIZERS.register_module(name=name, module=module)


register_jittor_optim()
