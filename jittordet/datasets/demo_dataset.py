from ..engine import DATASETS
from .base import BaseDetDataset


@DATASETS.register_module()
class DemoDataset(BaseDetDataset):

    def load_data_list(self):
        data = [
            dict(width=2, height=1),
            dict(width=1, height=2),
            dict(width=2, height=1),
            dict(width=2, height=1),
            dict(width=1, height=2),
            dict(width=2, height=1),
            dict(width=2, height=1),
            dict(width=1, height=2),
            dict(width=2, height=1),
            dict(width=2, height=1),
            dict(width=1, height=2),
            dict(width=2, height=1),
            dict(width=2, height=1),
            dict(width=1, height=2),
        ]
        return data
