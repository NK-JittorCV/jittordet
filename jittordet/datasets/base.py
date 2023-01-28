import copy
import os.path as osp

import jittor as jt
import numpy as np
from jittor.dataset import Dataset

from jittordet.engine import BATCH_SAMPLERS, TRANSFORMS
from jittordet.structures import DetDataSample, InstanceData


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


class BaseDetDataset(Dataset):

    METAINFO = dict()

    def __init__(self,
                 batch_size,
                 num_workers=0,
                 data_root='',
                 data_path=None,
                 metainfo=None,
                 filter_cfg=None,
                 test_mode=False,
                 transforms=None,
                 batch_sampler=None,
                 max_refetch=100,
                 **kwargs):
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, **kwargs)
        self.data_root = data_root
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self.test_mode = test_mode
        self.max_refetch = max_refetch

        # load metainfo
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        # datasetwise init
        self.init_datasetwise(data_path)
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

            def _join_data_root(value, data_root):
                if isinstance(value, list):
                    for i in range(len(value)):
                        value[i] = _join_data_root(value[i], data_root)
                elif isinstance(value, dict):
                    for key in value.keys():
                        value[key] = _join_data_root(value[key], data_root)
                elif isinstance(value, str):
                    if not osp.isabs(value):
                        value = osp.join(data_root, value)
                else:
                    raise TypeError(
                        'The contents in data_path should be str type.')
                return value

            v = _join_data_root(v, self.data_root)

            setattr(self, k, v)

    def load_data_list(self):
        raise NotImplementedError

    def filter_data(self):
        return self.data_list

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self.data_list) + idx

        data = self.transforms(data_info)
        return data

    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
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
            index = self.batch_sampler.get_index_list(rng=self._shuffle_rng)
            self.real_len = len(index)
            self.batch_len = len(self.batch_sampler)
            world_size = 1 if not jt.in_mpi else jt.world_size
            self.real_batch_size = int(self.batch_size // world_size)
            return index
        else:
            return super()._get_index_list()

    def collate_batch(self, batch):
        """Override original `collate_batch` to disable stack."""
        if isinstance(batch[0], dict):
            new_batch = dict()
            for key in batch[0].keys():
                value = [data[key] for data in batch]
                value = self.collate_batch(value)
                new_batch[key] = value
        elif isinstance(batch[0], list):
            new_batch = list()
            for i in range(len(batch[0])):
                value = [data[i] for data in batch]
                value = self.collate_batch(value)
                new_batch.append(value)
        elif isinstance(batch[0], tuple):
            new_batch = list()
            for i in range(len(batch[0])):
                value = [data[i] for data in batch]
                value = self.collate_batch(value)
                new_batch.append(value)
            new_batch = tuple(new_batch)
        else:
            new_batch = batch
        return new_batch

    def to_jittor(self, batch):
        """Override to_jittor function in jittor.utils.dataset."""
        if self.keep_numpy_array:
            return batch
        if isinstance(batch, jt.Var):
            return batch
        if isinstance(batch, (InstanceData, DetDataSample)):
            return batch.to_jittor(self.stop_grad)
        if isinstance(batch, np.ndarray):
            batch = jt.array(batch)
            if self.stop_grad:
                batch = batch.stop_grad()
        if isinstance(batch, dict):
            return {k: self.to_jittor(v) for k, v in batch.items()}
        if isinstance(batch, list):
            return [self.to_jittor(v) for v in batch]
        if isinstance(batch, tuple):
            batch = [self.to_jittor(v) for v in batch]
            return tuple(batch)
        return batch
