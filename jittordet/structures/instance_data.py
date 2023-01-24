# Modified from OpenMMLab mmengine.structures.instance_data.py
# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from collections.abc import Sized
from typing import List, Optional

import jittor as jt
import numpy as np

from .base_data_element import BaseDataElement


class InstanceData(BaseDataElement):

    def __setattr__(self, name: str, value: Sized):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'

            if len(self) > 0:
                assert len(value) == len(self), 'The length of ' \
                                                f'values {len(value)} is ' \
                                                'not consistent with ' \
                                                'the length of this ' \
                                                ':obj:`InstanceData` ' \
                                                f'{len(self)}'
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item) -> 'InstanceData':
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.
        Returns:
            :obj:`InstanceData`: Corresponding values.
        """
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = jt.array(item)

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, jt.Var):
            assert item.ndim == 1, 'Only support to get the' \
                                   ' values along the first dimension.'

            for k, v in self.items():
                if isinstance(v, jt.Var):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from BoolTensor
                    if item.dtype == jt.bool:
                        indexes = jt.nonzero(item).view(-1).numpy().tolist()
                    else:
                        indexes = item.numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        'support slice with `bool`')

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    @staticmethod
    def cat(instances_list: List['InstanceData']) -> 'InstanceData':
        """Concat the instances of all :obj:`InstanceData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.
        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.
        Returns:
            :obj:`InstanceData`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in instances_list
        ]
        assert len({len(field_keys) for field_keys in field_keys_list}) \
               == 1 and len(set(itertools.chain(*field_keys_list))) \
               == len(field_keys_list[0]), 'There are different keys in ' \
                                           '`instances_list`, which may ' \
                                           'cause the cat operation ' \
                                           'to fail. Please make sure all ' \
                                           'elements in `instances_list` ' \
                                           'have the exact same key.'

        new_data = instances_list[0].__class__(
            metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, jt.Var):
                new_values = jt.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, 'cat'):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f'The type of `{k}` is `{type(v0)}` which has no '
                    'attribute of `cat`')
            new_data[k] = new_values
        return new_data  # type:ignore

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0


InstanceList = List[InstanceData]
OptInstanceList = Optional[InstanceList]
