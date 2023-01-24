# Modified from OpenMMLab mmdet/models/data_preprocessors/data_preprocessor.py
# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
from typing import List, Optional, Sequence, Union

import jittor as jt
import jittor.nn as nn
import numpy as np

from jittordet.engine import MODELS
from jittordet.structures import DetDataSample
from jittordet.utils import is_list_of


@MODELS.register_module()
class Preprocessor(nn.Module):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to keep the ``BaseBoxes`` type of
            bboxes data or not. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__()
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or tuple')
        if mean is not None:
            assert len(mean) == 3 or len(mean) == 1, (
                '`mean` should have 1 or 3 values, to be compatible with '
                f'RGB or gray image, but got {len(mean)} values')
            assert len(std) == 3 or len(std) == 1, (  # type: ignore
                '`std` should have 1 or 3 values, to be compatible with RGB '  # type: ignore # noqa: E501
                f'or gray image, but got {len(std)} values')  # type: ignore
            self._enable_normalize = True
            self.mean = np.array(mean).reshape(-1, 1, 1)
            self.std = np.array(std).reshape(-1, 1, 1)
        else:
            self._enable_normalize = False
        self._channel_conversion = rgb_to_bgr or bgr_to_rgb
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value
        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None
        self.pad_mask = pad_mask
        self.mask_pad_value = mask_pad_value

    def execute(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_list_of(_batch_inputs, jt.Var):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert (
                            _batch_input.ndim == 3
                            and _batch_input.shape[0] == 3
                        ), ('If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    mean, std = jt.array(self.mean), jt.array(self.std)
                    _batch_input = (_batch_input - mean) / std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = self.stack_batch(batch_inputs,
                                            self.pad_size_divisor,
                                            self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, jt.Var):
            assert _batch_inputs.ndim == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                mean, std = jt.array(self.mean), jt.array(self.std)
                _batch_inputs = (_batch_inputs - mean) / std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = nn.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                  'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')

        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)

        inputs, data_samples = data['inputs'], data['data_samples']
        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    @staticmethod
    def stack_batch(tensor_list: List[jt.Var],
                    pad_size_divisor: int = 1,
                    pad_value: Union[int, float] = 0) -> jt.Var:
        """Stack multiple tensors to form a batch and pad the tensor to the max
        shape use the right bottom padding mode in these images. If
        ``pad_size_divisor > 0``, add padding to ensure the shape of each dim
        is divisible by ``pad_size_divisor``.

        Args:
            tensor_list (List[Tensor]): A list of tensors with the same dim.
            pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
                to ensure the shape of each dim is divisible by
                ``pad_size_divisor``. This depends on the model, and many
                models need to be divisible by 32. Defaults to 1
            pad_value (int, float): The padding value. Defaults to 0.

        Returns:
        Tensor: The n dim tensor.
        """
        assert isinstance(tensor_list, list), (
            f'Expected input type to be list, but got {type(tensor_list)}')
        assert tensor_list, '`tensor_list` could not be an empty list'
        assert len({
            tensor.ndim
            for tensor in tensor_list
        }) == 1, (f'Expected the dimensions of all tensors must be the same, '
                  f'but got {[tensor.ndim for tensor in tensor_list]}')

        dim = tensor_list[0].ndim
        num_img = len(tensor_list)
        all_sizes = jt.array([tensor.shape for tensor in tensor_list])
        max_sizes = jt.ceil(
            jt.max(all_sizes, dim=0) / pad_size_divisor) * pad_size_divisor
        padded_sizes = max_sizes - all_sizes
        # The first dim normally means channel,  which should not be padded.
        padded_sizes[:, 0] = 0
        if padded_sizes.sum() == 0:
            return jt.stack(tensor_list)
        # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
        # it means that padding the last dim with 1(left) 2(right), padding the
        # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite
        # of the `padded_sizes`. Therefore, the `padded_sizes` needs to be
        # reversed, and only odd index of pad should be assigned to keep
        # padding "right" and "bottom".
        pad = jt.zeros(num_img, 2 * dim, dtype=jt.int)
        index = list(range(dim - 1, -1, -1))
        pad[:, 1::2] = padded_sizes[:, index]
        batch_tensor = []
        for idx, tensor in enumerate(tensor_list):
            batch_tensor.append(
                nn.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
        return jt.stack(batch_tensor)

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_list_of(_batch_inputs, jt.Var):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, jt.Var):
            assert _batch_inputs.ndim == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a list of dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape

    def pad_gt_masks(self,
                     batch_data_samples: Sequence[DetDataSample]) -> None:
        """Pad gt_masks to shape of batch_input_shape."""
        if 'masks' in batch_data_samples[0].gt_instances:
            for data_samples in batch_data_samples:
                masks = data_samples.gt_instances.masks
                data_samples.gt_instances.masks = masks.pad(
                    data_samples.batch_input_shape,
                    pad_val=self.mask_pad_value)
