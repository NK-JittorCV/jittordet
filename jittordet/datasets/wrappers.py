from ..engine import BATCH_SAMPLERS, DATASETS
from ..utils import is_list_of
from .base import BaseDetDataset


class PartCompose:

    def __init__(self, composes, partition):
        assert len(composes) == len(partition)
        self.partition = partition
        self.composes = composes

    def __call__(self, data):
        assert 'sample_idx' in data
        sample_idx = data['sample_idx']
        for compose, part in zip(self.composes, self.partition):
            if sample_idx >= part:
                sample_idx -= part
                continue

            return compose(data)


@DATASETS.register_module()
class ConcatDataset(BaseDetDataset):

    def __init__(self,
                 datasets,
                 batch_size,
                 num_workers=0,
                 metainfo=None,
                 test_mode=False,
                 batch_sampler=None,
                 max_refetch=100,
                 **kwargs):
        super(BaseDetDataset, self).__init__(
            batch_size=batch_size, num_workers=num_workers, **kwargs)

        # override some setting in sub dataset
        assert is_list_of(datasets, dict)
        self.data_list, self.lengths = [], []
        transforms = []
        for dataset in datasets:
            dataset['batch_size'] = 1
            dataset['num_workers'] = 0
            dataset['metainfo'] = metainfo
            dataset['test_mode'] = test_mode
            dataset['batch_sampler'] = None
            dataset = DATASETS.build(dataset)
            self.data_list.extend(dataset.data_list)
            self.lengths.append(len(dataset.data_list))
            transforms.append(dataset.transforms)

        self.transforms = PartCompose(transforms, self.lengths)

        self.test_mode = test_mode
        self.max_refetch = max_refetch

        # set total length for jittor.utils.dataset
        self.total_len = len(self.data_list)

        if batch_sampler is not None:
            self.batch_sampler = BATCH_SAMPLERS.build(
                batch_sampler, dataset=self)
        else:
            self.batch_sampler = None
