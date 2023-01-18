from math import ceil

import jittor as jt
import numpy as np

from jittordet.engine import BATCH_SAMPLERS
from .base_batch_sampler import BaseBatchSampler


@BATCH_SAMPLERS.register_module()
class PadBatchSampler(BaseBatchSampler):

    def __init__(self, dataset, shuffle=False, drop_last=False):
        super().__init__(dataset=dataset)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        return self.batch_len

    @property
    def batch_len(self):
        length = self.data_list_len // self.batch_size
        if not self.drop_last:
            length += 1
        return length

    def get_index_list(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        index = rng.permutation(self.data_list_len) if self.shuffle \
            else np.arange(self.data_list_len)

        mod_size = self.data_list_len % self.batch_size
        if mod_size != 0:
            if self.drop_last:
                index = index[:-mod_size]
            else:
                padded_size = ceil(
                    self.data_list_len / self.batch_size) * self.batch_size
                # repeat index to avoid data_list is shorter than batch size
                repeat_num = int((self.batch_size - 0.5) //
                                 self.data_list_len + 1)
                repeat_num = max(repeat_num, 2)
                index = np.concatenate([index] * repeat_num)
                index = index[:padded_size]

        if jt.in_mpi:
            rank, world_size = jt.rank, jt.world_size
            real_bs = self.batch_size // world_size
            assert real_bs * world_size == self.batch_size
            index = index.reshape(-1, self.batch_size)
            index = index[:, rank * real_bs:(rank + 1) * real_bs]
            index = index.flatten()
            self.dataset.real_batch_size = real_bs
        else:
            self.dataset.real_batch_size = self.batch_size

        self.dataset.real_len = len(index)
        self.dataset.batch_len = self.batch_len
        return index
