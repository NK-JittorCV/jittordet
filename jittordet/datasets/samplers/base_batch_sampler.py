from abc import ABCMeta, abstractmethod, abstractproperty

from jittordet.engine import BATCH_SAMPLERS


@BATCH_SAMPLERS.register_module()
class BaseBatchSampler(metaclass=ABCMeta):

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def data_list_len(self):
        return len(self.dataset.data_list)

    @property
    def batch_size(self):
        return self.dataset.batch_size

    @abstractproperty
    def batch_len(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_index_list(self, rng=None):
        pass
