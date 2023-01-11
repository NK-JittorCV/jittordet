import os.path as osp
import numpy as np
import warnings

from jittordet.engine import DATASETS
from jittor.dataset import Dataset
from .piplines.transforms import Compose

@DATASETS.register_module()
class BaseDetDataset(Dataset):
    """Base dataset for JittorDet.
    """
    
    CLASSES = None
    
    def __init__(self,
                 ann_file,
                 transforms,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 batch_size=1,
                 num_workers=0,
                 shuffle=False,
                 drop_last=False,):
        super(BaseDetDataset,self).__init__(batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=shuffle,
                                           drop_last=drop_last)
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.CLASSES = self.get_classes(classes)
        
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)
        
        # load annotations
        self.data_infos = self.load_annotations(self.ann_file)
        
        # load proposals
        if self.proposal_file is not None:
            self.proposals = self.load_proposals(self.proposal_file)
        else:
            self.proposals = None
        
        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()
        
        self.total_len = len(self.data_infos)
        self.transforms = Compose(transforms)
        
        
    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        raise NotImplementedError


    def load_proposals(self, proposal_file):
        """Load proposal from proposal file."""
        raise NotImplementedError
    
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None):
        """Evaluation in COCO protocol."""
        raise NotImplementedError
    
    
    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']
    
    
    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()
    
    
    def pre_transforms(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
    
    
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    
    
    def collate_batch(self, batch):
        new_batch = {key:[] for key in batch[0]}
        max_width = 0
        max_height = 0
        for data in batch:
            for key in new_batch:
                if 'img' == key:
                    height,width = data['img'].shape[-2], data['img'].shape[-1]
                    max_width = max(max_width,width)
                    max_height = max(max_height,height)
                new_batch[key].append(data[key])                
        N = len(new_batch['img'])
        batch_imgs = np.zeros((N,3,max_height,max_width),dtype=np.float32)
        for i,img in enumerate(new_batch['img']):
            batch_imgs[i,:,:img.shape[-2],:img.shape[-1]] = img
        new_batch['img'] = batch_imgs
        return new_batch
    
    
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_transforms(results)
        return self.transforms(results)


    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_transforms(results)
        return self.transforms(results)
    
    
    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES
        
        if isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names
    
    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self.data_infos), dtype=np.uint8)
        for i in range(len(self.data_infos)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1
    
    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool) 