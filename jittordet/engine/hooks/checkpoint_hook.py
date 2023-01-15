from ..register import HOOKS
from .base_hook import BaseHook


@HOOKS.register_module()
class CheckpointHook(BaseHook):
    pass
