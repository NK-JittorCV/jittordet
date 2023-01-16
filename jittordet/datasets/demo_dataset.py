import jittor as jt
from jittor.dataset import Dataset

from jittordet.engine import DATASETS


@DATASETS.register_module()
class DemoDataset(Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_len = 1000

    def __getitem__(self, idx):
        return dict(data=jt.zeros([1, 256, 5, 5]))
