import numpy as np
import os.path as osp
from PIL import Image

# @PIPELINES.register_module()
class LoadImageFromFile:
    """Load an image from file.
    """
    def __init__(self,
                 to_float32=False,
                 image_colors=256,
                 mode='RGB'):
        self.to_float32 = to_float32
        self.image_colors = image_colors
        self.image_mode = mode

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = Image.open(filename).convert(mode=self.image_mode, colors=self.image_colors)
        img = np.array(img)
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.image_colors}', "
                    f"channel_order='{self.image_mode}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str

    
# @PIPELINES.register_module()
class LoadAnnotations:
    """Load multiple types of annotations.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 denorm_bbox=False):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.denorm_bbox = denorm_bbox

    def _load_bboxes(self, results):
        """Private function to load bounding box annotations.
        """

        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes'].copy()

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore.copy()
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')

        gt_is_group_ofs = ann_info.get('gt_is_group_ofs', None)
        if gt_is_group_ofs is not None:
            results['gt_is_group_ofs'] = gt_is_group_ofs.copy()

        return results

    def _load_labels(self, results):
        """Private function to load label annotations.
        """

        results['gt_labels'] = results['ann_info']['labels'].copy()
        return results


    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        return repr_str