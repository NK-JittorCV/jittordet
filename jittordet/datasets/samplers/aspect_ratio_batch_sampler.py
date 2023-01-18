import jittor as jt
import numpy as np

from jittordet.engine import BATCH_SAMPLERS
from .base_batch_sampler import BaseBatchSampler


@BATCH_SAMPLERS.register_module()
class AspectRatioBatchSampler(BaseBatchSampler):

    def __len__(self):
        return self.batch_len

    @property
    def batch_len(self):
        return self.data_list_len // self.batch_size

    def get_index_list(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        aspect_ratio_idx1 = []
        aspect_ratio_idx2 = []
        for idx, data in enumerate(self.dataset.data_list):
            if data['width'] > data['height']:
                aspect_ratio_idx1.append(idx)
            else:
                aspect_ratio_idx2.append(idx)
        aspect_ratio_idx1 = np.array(aspect_ratio_idx1)
        aspect_ratio_idx2 = np.array(aspect_ratio_idx2)

        # shuffle
        shuffle_idx = rng.permutation(aspect_ratio_idx1.size)
        aspect_ratio_idx1 = aspect_ratio_idx1[shuffle_idx]
        shuffle_idx = rng.permutation(aspect_ratio_idx2.size)
        aspect_ratio_idx2 = aspect_ratio_idx2[shuffle_idx]

        # drop last size
        world_size = 1 if not jt.in_mpi else jt.world_size
        real_bs = self.batch_size // world_size
        assert real_bs * jt.world_size == self.batch_size
        if aspect_ratio_idx1.size % real_bs != 0:
            mod_size = aspect_ratio_idx1.size % real_bs
            aspect_ratio_idx1 = aspect_ratio_idx1[:-mod_size]
        aspect_ratio_idx1 = aspect_ratio_idx1.reshape(-1, real_bs)
        if aspect_ratio_idx2.size % real_bs != 0:
            mod_size = aspect_ratio_idx2.size % real_bs
            aspect_ratio_idx2 = aspect_ratio_idx2[:-mod_size]
        aspect_ratio_idx2 = aspect_ratio_idx2.reshape(-1, real_bs)

        index = np.concatenate([aspect_ratio_idx1, aspect_ratio_idx2], axis=0)
        shuffle_idx = rng.permutation(index.shape[0])
        index = index[shuffle_idx]

        real_bs_num = self.batch_len * world_size
        repeat_num = (index.shape[0] - 1) // real_bs_num + 1
        index = np.concatenate([index] * repeat_num, axis=0)
        index = index[:real_bs_num]

        if jt.in_mpi:
            index = index.reshape(-1, self.batch_size)
            index = index[:, jt.rank * real_bs:(jt.rank + 1) * real_bs]

        index = index.flatten()
        self.dataset.real_len = len(index)
        self.dataset.batch_len = self.batch_len
        self.dataset.real_batch_size = real_bs
        return index
