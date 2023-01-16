from abc import ABCMeta, abstractmethod


class BaseLoop(metaclass=ABCMeta):

    def __init__(self, runner):
        self._runner = runner
        self._max_epoch = 1
        self._epoch = 0
        self._iter = 0

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

    def state_dict(self):
        return dict(epoch=self._epoch, iter=self._iter)

    def load_state_dict(self, data):
        assert isinstance(data, dict)
        self._epoch = data['epoch']
        self._iter = data['iter']
