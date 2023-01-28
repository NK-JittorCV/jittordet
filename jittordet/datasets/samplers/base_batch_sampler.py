from abc import ABCMeta, abstractmethod

from jittordet.engine import BATCH_SAMPLERS


@BATCH_SAMPLERS.register_module()
class BaseBatchSampler(metaclass=ABCMeta):

    def __init__(self, dataset):
        self.total_bs = dataset.batch_size
        self.num_data_list = len(dataset.data_list)

    def __len__(self):
        length = int((self.num_data_list - 0.5) // self.total_bs)
        if hasattr(self, 'drop_last') and not self.drop_last:
            length += 1
        return length

    @abstractmethod
    def get_index_list(self, rng=None):
        pass
