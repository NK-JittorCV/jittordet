# Modified from OpenMMLab.
# mmdet/models/task_modules/samplers/sampling_result.py
# Copyright (c) OpenMMLab. All rights reserved.
import jittor as jt

from ..assigners import AssignResult


class SamplingResult:
    """Bbox sampling result.

    Args:
        pos_inds (Tensor): Indices of positive samples.
        neg_inds (Tensor): Indices of negative samples.
        priors (Tensor): The priors can be anchors or points,
            or the bboxes predicted by the previous stage.
        gt_bboxes (Tensor): Ground truth of bboxes.
        assign_result (:obj:`AssignResult`): Assigning results.
        gt_flags (Tensor): The Ground truth flags.
        avg_factor_with_neg (bool):  If True, ``avg_factor`` equal to
            the number of total priors; Otherwise, it is the number of
            positive priors. Defaults to True.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.models.task_modules.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_inds': tensor([1,  2,  3,  5,  6,  7,  8,
                                9, 10, 11, 12, 13]),
            'neg_priors': torch.Size([12, 4]),
            'num_gts': 1,
            'num_neg': 12,
            'num_pos': 1,
            'avg_factor': 13,
            'pos_assigned_gt_inds': tensor([0]),
            'pos_inds': tensor([0]),
            'pos_is_gt': tensor([1], dtype=torch.uint8),
            'pos_priors': torch.Size([1, 4])
        })>
    """

    def __init__(self,
                 pos_inds: jt.Var,
                 neg_inds: jt.Var,
                 priors: jt.Var,
                 gt_bboxes: jt.Var,
                 assign_result: AssignResult,
                 gt_flags: jt.Var,
                 avg_factor_with_neg: bool = True) -> None:
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.num_pos = max(pos_inds.numel(), 1)
        self.num_neg = max(neg_inds.numel(), 1)
        self.avg_factor_with_neg = avg_factor_with_neg
        self.avg_factor = self.num_pos + self.num_neg \
            if avg_factor_with_neg else self.num_pos
        self.pos_priors = priors[pos_inds]
        self.neg_priors = priors[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_labels = assign_result.labels[pos_inds]
        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_bboxes = gt_bboxes.view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long()]

    @property
    def priors(self):
        """torch.Tensor: concatenated positive and negative priors"""
        return jt.concat([self.pos_priors, self.neg_priors])

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_priors': self.pos_priors,
            'neg_priors': self.neg_priors,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
            'num_pos': self.num_pos,
            'num_neg': self.num_neg,
            'avg_factor': self.avg_factor
        }
