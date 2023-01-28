import jittor as jt
import numpy as np

from jittordet.engine import BATCH_SAMPLERS
from .base_batch_sampler import BaseBatchSampler


@BATCH_SAMPLERS.register_module()
class AspectRatioBatchSampler(BaseBatchSampler):

    def __init__(self, dataset, drop_last=False):
        super().__init__(dataset=dataset)
        self.drop_last = drop_last
        # statstic different aspect ratios idx.
        idx_bucket1, idx_bucket2 = [], []
        for idx, data in enumerate(dataset.data_list):
            if data['width'] > data['height']:
                idx_bucket1.append(idx)
            else:
                idx_bucket2.append(idx)
        self.idx_bucket1 = np.array(idx_bucket1)
        self.idx_bucket2 = np.array(idx_bucket2)

    def get_index_list(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        # shuffle
        shuffle_idx = rng.permutation(self.idx_bucket1.size)
        idx_bucket1 = self.idx_bucket1[shuffle_idx]
        shuffle_idx = rng.permutation(self.idx_bucket2.size)
        idx_bucket2 = self.idx_bucket2[shuffle_idx]

        # drop last size
        world_size = 1 if not jt.in_mpi else jt.world_size
        total_bs = self.total_bs
        real_bs = int(total_bs // world_size)
        assert real_bs * jt.world_size == total_bs
        if idx_bucket1.size % real_bs != 0:
            mod_size = idx_bucket1.size % real_bs
            idx_bucket1 = idx_bucket1[:-mod_size]
        idx_bucket1 = idx_bucket1.reshape(-1, real_bs)
        if idx_bucket2.size % real_bs != 0:
            mod_size = idx_bucket2.size % real_bs
            idx_bucket2 = idx_bucket2[:-mod_size]
        idx_bucket2 = idx_bucket2.reshape(-1, real_bs)

        index = np.concatenate([idx_bucket1, idx_bucket2], axis=0)
        shuffle_idx = rng.permutation(index.shape[0])
        index = index[shuffle_idx]

        real_bs_num = len(self) * world_size
        repeat_num = int((real_bs_num - 0.5) // index.shape[0] + 1)
        index = np.concatenate([index] * repeat_num, axis=0)
        index = index[:real_bs_num]

        if jt.in_mpi:
            index = index.reshape(-1, total_bs)
            index = index[:, jt.rank * real_bs:(jt.rank + 1) * real_bs]

        index = index.flatten()
        return index
