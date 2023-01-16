import numpy as np

from jittordet.engine import TRANSFORM


@TRANSFORM.register_module()
class DefaultFormatBundle:
    """Default formatting bundle."""

    def __init__(self,
                 img_to_float=True,
                 pad_val=dict(img=0, masks=0, seg=255)):
        self.img_to_float = img_to_float
        self.pad_val = pad_val

    def __call__(self, data: dict) -> dict:
        """Call function to transform and format common fields in results."""
        if 'img' in data:
            img = data['img']
            if self.img_to_float is True and img.dtype == np.uint8:
                # Normally, image is of uint8 type without normalization.
                # At this time, it needs to be forced to be converted to
                # flot32, otherwise the model training and inference
                # will be wrong. Only used for YOLOX currently .
                img = img.astype(np.float32)
            # add default meta keys
            data = self._add_default_meta_keys(data)
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            data['img'] = img
        return data

    def _add_default_meta_keys(self, data: dict) -> dict:
        """Add default meta keys."""
        img = data['img']
        data.setdefault('pad_shape', img.shape)
        data.setdefault('scale_factor', 1.0)
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        data.setdefault(
            'img_norm_cfg',
            dict(
                mean=np.zeros(num_channels, dtype=np.float32),
                std=np.ones(num_channels, dtype=np.float32),
                to_rgb=False))
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(img_to_float={self.img_to_float})'


@TRANSFORM.register_module()
class Collect:
    """Collect data from the loader relevant to the specific task."""

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, data: dict) -> dict:
        """Call function to collect keys in results."""
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = data[key]
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = data[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
               f'(keys={self.keys}, meta_keys={self.meta_keys})'


@TRANSFORM.register_module()
class WrapFieldsToLists:
    """Wrap fields of the data dictionary into lists for evaluation."""

    def __call__(self, data: dict) -> dict:
        """Call function to wrap fields into lists."""
        # Wrap dict fields into lists
        for key, val in data.items():
            data[key] = [val]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'
