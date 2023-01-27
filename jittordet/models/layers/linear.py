import jittor.nn as nn

from jittordet.engine import MODELS

MODELS.register_module('Linear', module=nn.Linear)
