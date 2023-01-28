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

    def get_index_list(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        index = rng.permutation(self.num_data_list) if self.shuffle \
            else np.arange(self.num_data_list)

        mod_size = self.num_data_list % self.total_bs
        if mod_size != 0:
            if self.drop_last:
                index = index[:-mod_size]
            else:
                padded_size = int(
                    ceil(self.num_data_list / self.total_bs) * self.total_bs)
                # repeat index to avoid data_list is shorter than batch size
                repeat_num = int((self.total_bs - 0.5) // self.num_data_list +
                                 1)
                repeat_num = max(repeat_num, 2)
                index = np.concatenate([index] * repeat_num)
                index = index[:padded_size]

        if jt.in_mpi:
            rank, world_size = jt.rank, jt.world_size
            real_bs = int(self.total_bs // world_size)
            assert real_bs * world_size == self.total_bs
            index = index.reshape(-1, self.total_bs)
            index = index[:, rank * real_bs:(rank + 1) * real_bs]
            index = index.flatten()

        return index
