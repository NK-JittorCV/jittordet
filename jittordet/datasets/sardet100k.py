# Modified from mmdetection.dataset.coco
from ..engine import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class Sardet100k(CocoDataset):
    """
        Dataset for Sardet100k.
        Download dataset at: https://liveuclac-my.sharepoint.com/:f:/g/personal/zcablii_ucl_ac_uk/EuYYZWXL_bJGvd8s9rGH2KYBV1GM5pIOCngnzlyuB_3e5A?e=bgoINm
    """
  
    METAINFO = {
        'classes':
        ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100)]
    }
