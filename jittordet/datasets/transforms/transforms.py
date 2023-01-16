import collections

import numpy as np

from jittordet.engine import TRANSFORM
from ..utils import (imflip, imnormalize, impad, impad_to_multiple, imrescale,
                     imresize)


@TRANSFORM.register_module()
class Resize:
    """Resize images & bbox & mask."""

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',
                 override=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert isinstance(self.img_scale[0], tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates."""
        assert isinstance(img_scales[0], tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``."""
        assert isinstance(img_scales[0], tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified."""
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, data: dict):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``."""

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        data['scale'] = scale
        data['scale_idx'] = scale_idx

    def _resize_img(self, data: dict):
        """Resize images with ``data['scale']``."""
        for key in data.get('img_fields', ['img']):
            if self.keep_ratio:
                img, scale_factor = imrescale(
                    data[key],
                    data['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = data[key].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = imresize(
                    data[key],
                    data['scale'],
                    return_scale=True,
                    interpolation=self.interpolation,
                    backend=self.backend)
            data[key] = img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
            data['img_shape'] = img.shape
            # in case that there is no padding
            data['pad_shape'] = img.shape
            data['scale_factor'] = scale_factor
            data['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, data: dict):
        """Resize bounding boxes with ``data['scale_factor']``."""
        for key in data.get('bbox_fields', []):
            bboxes = data[key] * data['scale_factor']
            if self.bbox_clip_border:
                img_shape = data['img_shape']
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            data[key] = bboxes

    def __call__(self, data: dict) -> dict:
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map."""

        if 'scale' not in data:
            if 'scale_factor' in data:
                img_shape = data['img'].shape[:2]
                scale_factor = data['scale_factor']
                assert isinstance(scale_factor, float)
                data['scale'] = tuple(
                    [int(x * scale_factor) for x in img_shape][::-1])
            else:
                self._random_scale(data)
        else:
            if not self.override:
                assert 'scale_factor' not in data, (
                    'scale and scale_factor cannot be both set.')
            else:
                data.pop('scale')
                if 'scale_factor' in data:
                    data.pop('scale_factor')
                self._random_scale(data)
        self._resize_img(data)
        self._resize_bboxes(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORM.register_module()
class RandomFlip:
    """Flip the image & bbox & mask."""

    def __init__(self, flip_ratio=None, direction='horizontal'):
        if isinstance(flip_ratio, list):
            for i in flip_ratio:
                assert isinstance(i, float)
            assert 0 <= sum(flip_ratio) <= 1
        elif isinstance(flip_ratio, float):
            assert 0 <= flip_ratio <= 1
        elif flip_ratio is None:
            pass
        else:
            raise ValueError('flip_ratios must be None, float, '
                             'or list of float')
        self.flip_ratio = flip_ratio

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            for i in direction:
                assert isinstance(i, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError('direction must be either str or list of str')
        self.direction = direction

        if isinstance(flip_ratio, list):
            assert len(self.flip_ratio) == len(self.direction)

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally."""
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            w = img_shape[1]
            h = img_shape[0]
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(f"Invalid flipping direction '{direction}'")
        return flipped

    def __call__(self, data):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps."""
        if 'flip' not in data:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]
            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            data['flip'] = cur_dir is not None
        if 'flip_direction' not in data:
            data['flip_direction'] = cur_dir
        if data['flip']:
            # flip image
            for key in data.get('img_fields', ['img']):
                data[key] = imflip(data[key], direction=data['flip_direction'])
            # flip bboxes
            for key in data.get('bbox_fields', []):
                data[key] = self.bbox_flip(data[key], data['img_shape'],
                                           data['flip_direction'])
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_ratio={self.flip_ratio})'


@TRANSFORM.register_module()
class Pad:
    """Pad the image & masks & segmentation map."""

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            pad_val = dict(img=pad_val)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square
        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def __call__(self, data: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps."""
        pad_val = self.pad_val.get('img', 0)
        for key in data.get('img_fields', ['img']):
            if self.pad_to_square:
                max_size = max(data[key].shape[:2])
                self.size = (max_size, max_size)
            if self.size is not None:
                padded_img = impad(data[key], shape=self.size, pad_val=pad_val)
            elif self.size_divisor is not None:
                padded_img = impad_to_multiple(
                    data[key], self.size_divisor, pad_val=pad_val)
            data[key] = padded_img
        data['pad_shape'] = padded_img.shape
        data['pad_fixed_size'] = self.size
        data['pad_size_divisor'] = self.size_divisor
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@TRANSFORM.register_module()
class Normalize:
    """Normalize the image."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, data):
        """Call function to normalize images."""
        for key in data.get('img_fields', ['img']):
            data[key] = imnormalize(data[key], self.mean, self.std,
                                    self.to_rgb)
        data['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@TRANSFORM.register_module()
class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                # TODO: transform = build_from_cfg(transform, TRANSFORM)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially."""
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string
