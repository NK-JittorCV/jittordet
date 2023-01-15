from abc import ABCMeta, abstractmethod


class BaseLoop(metaclass=ABCMeta):

    def __init__(self, runner):
        self._runner = runner

    @property
    def runner(self):
        return self._runner

    @property
    def cur_epoch(self):
        return self._epoch

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def cur_iter(self):
        return self._iter

    @abstractmethod
    def run(self):
        pass
