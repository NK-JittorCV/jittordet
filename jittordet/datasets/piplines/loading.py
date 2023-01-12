import numpy as np
import os.path as osp
from PIL import Image

from jittordet.engine import TRANSFORM

@TRANSFORM.register_module()
class LoadImageFromFile:
    """Load an image from file."""
    def __init__(self,
                 to_float32=False,
                 image_colors=256,
                 mode='RGB'):
        self.to_float32 = to_float32
        self.image_colors = image_colors
        self.image_mode = mode

    def __call__(self, data: dict) -> dict:
        """Call functions to load image and get image meta information."""
        if data['img_prefix'] is not None:
            filename = osp.join(data['img_prefix'],
                                data['img_info']['filename'])
        else:
            filename = data['img_info']['filename']

        img = np.array(Image.open(filename).convert(mode=self.image_mode, colors=self.image_colors))
        
        if self.to_float32:
            img = img.astype(np.float32)

        data['filename'] = filename
        data['ori_filename'] = data['img_info']['filename']
        data['img'] = img
        data['img_shape'] = img.shape
        data['ori_shape'] = img.shape
        data['img_fields'] = ['img']
        return data

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"image_colors='{self.image_colors}', "
                    f'image_mode={self.image_mode})')
        return repr_str

    
@TRANSFORM.register_module()
class LoadAnnotations:
    """Load multiple types of annotations."""
    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 denorm_bbox=False):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.denorm_bbox = denorm_bbox

    def _load_bboxes(self, data: dict) -> dict:
        """Private function to load bounding box annotations."""
        ann_info = data['ann_info']
        data['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            bbox_num = data['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = data['img_shape'][:2]
                data['gt_bboxes'][:, 0::2] *= w
                data['gt_bboxes'][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            data['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            data['bbox_fields'].append('gt_bboxes_ignore')
        data['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            data['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return data

    def _load_labels(self, data: dict) -> dict:
        """Private function to load label annotations."""
        data['gt_labels'] = data['ann_info']['labels'].copy()
        return data


    def __call__(self, data: dict) -> dict:
        """Call function to load multiple types annotations."""
        if self.with_bbox:
            data = self._load_bboxes(data)
            if data is None:
                return None
        if self.with_label:
            data = self._load_labels(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label})'
        return repr_str