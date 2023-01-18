# modified from mmengine.dataset.BaseDataset

import copy
import os.path as osp

import numpy as np
from jittor.dataset import Dataset

from jittordet.engine import BATCH_SAMPLERS, DATASETS, TRANSFORMS


class Compose:
    """Modified from mmengine.dataset.base_dataest.

    Compose multiple transforms sequentially.
    """

    def __init__(self, transforms=None):
        if transforms is None:
            self.transforms = []
            return

        # validate data type
        if isinstance(transforms, dict):
            transforms = [transforms]

        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data


@DATASETS.register_module()
class BaseDetDataset(Dataset):

    METAINFO = dict()

    def __init__(self,
                 batch_size,
                 num_works=0,
                 data_root='',
                 datasetwise_cfg=None,
                 metainfo=None,
                 filter_cfg=None,
                 test_mode=False,
                 transforms=None,
                 batch_sampler=None,
                 max_refetch=100,
                 **kwargs):
        super().__init__(
            batch_size=batch_size, num_workers=num_works, **kwargs)
        self.data_root = data_root
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self.test_mode = test_mode
        self.max_refetch = max_refetch

        # load metainfo
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        # datasetwise init
        self.init_datasetwise(datasetwise_cfg)
        # load data information
        self.data_list = self.load_data_list()
        # fliter illegal data
        self.data_list = self.filter_data()

        # set total length for jittor.utils.dataset
        self.total_len = len(self.data_list)

        # compose data transforms
        self.transforms = Compose(transforms)

        if batch_sampler is not None:
            self.batch_sampler = BATCH_SAMPLERS.build(
                batch_sampler, dataset=self)
        else:
            self.batch_sampler = None

    @property
    def metainfo(self):
        return copy.deepcopy(self._metainfo)

    @classmethod
    def _load_metainfo(cls, metainfo):
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, dict):
            raise TypeError(
                f'metainfo should be a dict, but got {type(metainfo)}')

        cls_metainfo.update(metainfo)
        return cls_metainfo

    def init_datasetwise(self, datasetwise_cfg):
        if datasetwise_cfg is None:
            return
        assert isinstance(datasetwise_cfg, dict), \
            f'datasetwise_cfg must be a dict, but get {type(datasetwise_cfg)}'

        for k, v in datasetwise_cfg.items():
            if hasattr(self, k):
                raise RuntimeError(f'Attr {k} has been set in {type(self)}')
            if 'path' in k and not osp.isabs(v):
                v = osp.join(self.data_root, v)
            setattr(self, k, v)

    def load_data_list(self):
        raise NotImplementedError

    def filter_data(self):
        return self.data_list

    def __getitem__(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self.data_list) + idx

        if self.test_mode:
            data = self.transforms(data_info)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.transforms(data_info)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = np.random.randint(0, len(self.data_list))
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    def __len__(self):
        if self.batch_sampler is not None:
            return len(self.batch_sampler)
        else:
            return super().__len__()

    def __batch_len__(self):
        if self.batch_sampler is not None:
            return self.batch_sampler.batch_len
        else:
            return super().__batch_len__()

    def _get_index_list(self):
        if self.batch_sampler is not None:
            return self.batch_sampler.get_index_list(rng=self._shuffle_rng)
        else:
            return super()._get_index_list()

    def collate_batch(self, batch):
        """Disable batch collating function in jittor.utils.dataset."""
        return batch