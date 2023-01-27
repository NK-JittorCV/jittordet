# Modified from OpenMMLab.
# mmdet/models/task_modules/samplers/random_sampler.py
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import jittor as jt
from numpy import ndarray

from jittordet.engine import TASK_UTILS
from ..assigners import AssignResult
from .base_sampler import BaseSampler


@TASK_UTILS.register_module()
class RandomSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num: int,
                 pos_fraction: float,
                 neg_pos_ub: int = -1,
                 add_gt_as_proposals: bool = True,
                 **kwargs):
        from jittordet.utils import ensure_rng
        super().__init__(
            num=num,
            pos_fraction=pos_fraction,
            neg_pos_ub=neg_pos_ub,
            add_gt_as_proposals=add_gt_as_proposals)
        self.rng = ensure_rng(kwargs.get('rng', None))

    def random_choice(self, gallery: Union[jt.Var, ndarray, list],
                      num: int) -> Union[jt.Var, ndarray]:
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, jt.Var)
        if not is_tensor:
            gallery = jt.array(gallery, dtype=jt.int64)
        # This is a temporary fix. We can revert the following code
        # when PyTorch fixes the abnormal return of torch.randperm.
        # See: https://github.com/open-mmlab/mmdetection/pull/5014
        perm = jt.randperm(gallery.numel())[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.numpy()
        return rand_inds

    def _sample_pos(self, assign_result: AssignResult, num_expected: int,
                    **kwargs) -> Union[jt.Var, ndarray]:
        """Randomly sample some positive samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        pos_inds = jt.nonzero(assign_result.gt_inds > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result: AssignResult, num_expected: int,
                    **kwargs) -> Union[jt.Var, ndarray]:
        """Randomly sample some negative samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = jt.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)
