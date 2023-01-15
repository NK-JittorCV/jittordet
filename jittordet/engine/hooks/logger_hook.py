from ..register import HOOKS
from .base_hook import BaseHook


@HOOKS.register_module()
class LoggerHook(BaseHook):

    def __init__(self):
        pass
