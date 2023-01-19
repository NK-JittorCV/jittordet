import cv2
import numpy as np

from jittordet.engine import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageFromFile:
    """Load an image from file."""

    def __init__(self, to_float32=False):
        self.to_float32 = to_float32

    def __call__(self, results):
        img = cv2.imread(results['img_path'])
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations:
    """Load multiple types of annotations."""

    def __init__(self, with_bbox=True, with_label=True):
        self.with_bbox = with_bbox
        self.with_label = with_label

    def _load_bboxes(self, results):
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])

        results['gt_bboxes'] = np.array(
            gt_bboxes, dtype=np.float32).reshape((-1, 4))
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=np.bool)

    def _load_labels(self, results):
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def __call__(self, results):
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label})'
        return repr_str
