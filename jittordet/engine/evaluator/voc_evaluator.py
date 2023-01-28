# Modified from OpenMMLab. mmdet/evaluation/metrics/voc_metric.py
# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import List, Optional, Sequence, Union

import numpy as np

from jittordet.structures import BaseDataElement
from ..register import EVALUATORS
from .base_evaluator import BaseEvaluator
from .utils import eval_map, eval_recalls


@EVALUATORS.register_module()
class VocEvaluator(BaseEvaluator):
    """Pascal VOC evaluation metric.

    Args:
        iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
        scale_ranges (List[tuple], optional): Scale ranges for evaluating
            mAP. If not specified, all bounding boxes would be included in
            evaluation. Defaults to None.
        metric (str | list[str]): Metrics to be evaluated. Options are
            'mAP', 'recall'. If is list, the first setting in the list will
             be used to evaluate metric.
        proposal_nums (Sequence[int]): Proposal number used for evaluating
            recalls, such as recall@100, recall@1000.
            Default: (100, 300, 1000).
        eval_mode (str): 'area' or '11points', 'area' means calculating the
            area under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1].
            The PASCAL VOC2007 defaults to use '11points', while PASCAL
            VOC2012 defaults to use 'area'.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    default_prefix: Optional[str] = 'pascal_voc'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 eval_mode: str = '11points') -> None:
        self.results = []
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        self.scale_ranges = scale_ranges
        # voc evaluation metrics
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['recall', 'mAP']
        if metric not in allowed_metrics:
            raise KeyError(
                f"metric should be one of 'recall', 'mAP', but got {metric}.")
        self.metric = metric
        self.proposal_nums = proposal_nums
        assert eval_mode in ['area', '11points'], \
            'Unrecognized mode, only "area" and "11points" are supported'
        self.eval_mode = eval_mode

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, dataset, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.
        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            if isinstance(data_sample, BaseDataElement):
                data_sample = data_sample.to_dict()

            pred = data_sample['pred_instances']
            pred_bboxes = pred['bboxes'].numpy()
            pred_scores = pred['scores'].numpy()
            pred_labels = pred['labels'].numpy()

            dets = []
            for label in range(len(dataset.metainfo['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            sample_idx = data_sample['sample_idx']
            gt_info = dataset.data_list[sample_idx]
            gt_labels, gt_bboxes = [], []
            gt_labels_ignore, gt_bboxes_ignore = [], []
            for ann in gt_info['instances']:
                if ann['ignore_flag']:
                    gt_bboxes_ignore.append(ann['bbox'])
                    gt_labels_ignore.append(ann['bbox_label'])
                else:
                    gt_bboxes.append(ann['bbox'])
                    gt_labels.append(ann['bbox_label'])
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32) \
                if gt_bboxes else np.zeros((0, 4), dtype=np.float32)
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32) \
                if gt_bboxes_ignore else np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64) \
                if gt_labels else np.zeros((0, ), dtype=np.int64)
            gt_labels_ignore = np.array(gt_labels_ignore, dtype=np.int64) \
                if gt_labels_ignore else np.zeros((0, ), dtype=np.int64)
            gt = dict(
                labels=gt_labels,
                bboxes=gt_bboxes,
                labels_ignore=gt_labels_ignore,
                bboxes_ignore=gt_bboxes_ignore)
            self.results.append((gt, dets))

    def compute_metrics(self, dataset, results: list, logger) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.
        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        gts, preds = zip(*results)
        eval_results = OrderedDict()
        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            if self.eval_mode == '11points':
                dataset_name = 'voc07'
            else:
                dataset_name = 'voc12'
            class_names = dataset.metainfo['classes']

            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                # Follow the official implementation,
                # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
                # we should use the legacy coordinate system in mmdet 1.x,
                # which means w, h should be computed as 'x2 - x1 + 1` and
                # `y2 - y1 + 1`
                mean_ap, _ = eval_map(
                    preds,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    dataset=dataset_name,
                    class_names=class_names,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif self.metric == 'recall':
            # TODO: Currently not checked.
            gt_bboxes = [ann['bboxes'] for ann in self.annotations]
            recalls = eval_recalls(
                gt_bboxes,
                results,
                self.proposal_nums,
                self.iou_thrs,
                logger=logger,
                use_legacy_coordinate=True)
            for i, num in enumerate(self.proposal_nums):
                for j, iou_thr in enumerate(self.iou_thrs):
                    eval_results[f'recall@{num}@{iou_thr}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(self.proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results
