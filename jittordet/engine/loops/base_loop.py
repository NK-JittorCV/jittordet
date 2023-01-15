from abc import ABCMeta, abstractmethod


class BaseLoop(metaclass=ABCMeta):

    def __init__(self, runner):
        self._runner = runner

    @property
    def runner(self):
        return self._runner

    @abstractmethod
    def run(self):
        pass
