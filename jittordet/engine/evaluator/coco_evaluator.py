# Modified from OpenMMLab. mmdet/evaluation/metrics/coco_metric.py
# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import json
import os
import os.path as osp
# import warnings
from collections import OrderedDict
# from typing import List, Optional, Sequence, Union
from typing import Optional

import numpy as np
# from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from ..register import EVALUATORS
from .base_evaluator import BaseEvaluator


@EVALUATORS.register_module()
class CocoEvaluator(BaseEvaluator):
    """COCO evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.
    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    default_prefix: Optional[str] = 'coco'

    # def __init__(self,
    #              ann_file: Optional[str] = None,
    #              metric: Union[str, List[str]] = 'bbox',
    #              classwise: bool = False,
    #              proposal_nums: Sequence[int] = (100, 300, 1000),
    #              iou_thrs: Optional[Union[float, Sequence[float]]] = None,
    #              metric_items: Optional[Sequence[str]] = None,
    #              format_only: bool = False,
    #              outfile_prefix: Optional[str] = None,
    #              file_client_args: dict = dict(backend='disk'),
    #              collect_device: str = 'cpu',
    #              prefix: Optional[str] = None) -> None:
    #     super().__init__(collect_device=collect_device, prefix=prefix)
    #     # coco evaluation metrics
    #     self.metrics = metric if isinstance(metric, list) else [metric]
    #     allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
    #     for metric in self.metrics:
    #         if metric not in allowed_metrics:
    #             raise KeyError(
    #                 "metric should be one of 'bbox', 'segm', 'proposal', "
    #                 f"'proposal_fast', but got {metric}.")

    #     # do class wise evaluation, default False
    #     self.classwise = classwise

    #     # proposal_nums used to compute recall or precision.
    #     self.proposal_nums = list(proposal_nums)

    #     # iou_thrs used to compute recall or precision.
    #     if iou_thrs is None:
    #         iou_thrs = np.linspace(
    #             .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1,
    #               endpoint=True)
    #     self.iou_thrs = iou_thrs
    #     self.metric_items = metric_items
    #     self.format_only = format_only
    #     if self.format_only:
    #         assert outfile_prefix is not None, 'outfile_prefix must be not'
    #         'None when format_only is True, otherwise the result files will'
    #         'be saved to a temp directory which will be cleaned up at the
    # end.'

    #     self.outfile_prefix = outfile_prefix

    #     self.file_client_args = file_client_args
    #     self.file_client = FileClient(**file_client_args)

    #     # if ann_file is not specified,
    #     # initialize coco api with the converted dataset
    #     if ann_file is not None:
    #         with self.file_client.get_local_path(ann_file) as local_path:
    #             self._coco_api = COCO(local_path)
    #     else:
    #         self._coco_api = None

    #     # handle dataset lazy init
    #     self.cat_ids = None
    #     self.img_ids = None

    # def __init__(self,
    #              ann_file: Optional[str] = None,
    #              metric: Union[str, List[str]] = 'bbox',
    #              classwise: bool = False,
    #              proposal_nums: Sequence[int] = (100, 300, 1000),
    #              iou_thrs: Optional[Union[float, Sequence[float]]] = None,
    #              metric_items: Optional[Sequence[str]] = None,
    #              format_only: bool = False,
    #              outfile_prefix: Optional[str] = None) -> None:
    #     pass
    # super().__init__()
    # # coco evaluation metrics
    # self.metrics = metric if isinstance(metric, list) else [metric]
    # allowed_metrics = ['bbox', 'proposal']
    # for metric in self.metrics:
    #     if metric not in allowed_metrics:
    #         raise KeyError(
    #             "metric should be one of 'bbox', 'segm', 'proposal', "
    #             f"'proposal_fast', but got {metric}.")

    # # do class wise evaluation, default False
    # self.classwise = classwise

    # # proposal_nums used to compute recall or precision.
    # self.proposal_nums = list(proposal_nums)

    # # iou_thrs used to compute recall or precision.
    # self.iou_thrs = None
    # if iou_thrs is None:
    #     self.iou_thrs = np.linspace(
    #         .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1,
    #         endpoint=True)

    # self.metric_items = metric_items
    # self.format_only = format_only
    # self.outfile_prefix = outfile_prefix

    # # if ann_file is not specified,
    # # initialize coco api with the converted dataset
    # if ann_file is not None:
    #     self._coco_api = COCO(ann_file)
    # else:
    #     self._coco_api = None

    # # handle dataset lazy init
    # self.cat_ids = None
    # self.img_ids = None

    def process_results(self, results):
        preds = []
        for result in results:
            pred = dict()
            result_ = result['pred_instances']
            pred['img_id'] = result['img_id']
            pred['bboxes'] = result_['bboxes'].cpu().numpy()
            pred['scores'] = result_['scores'].cpu().numpy()
            pred['labels'] = result_['labels'].cpu().numpy()
            preds.append(pred)
        return preds

    def results2json(self, results, save_file):
        """Convert detection results to COCO json style."""

        def xyxy2xywh(box):
            x1, y1, x2, y2 = box.tolist()
            return [x1, y1, x2 - x1, y2 - y1]

        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)
        json.dump(bbox_json_results, save_file)

    def build_file(self, epoch):
        os.makedirs(self.outfile_prefix, exist_ok=True)
        filename = 'val_{}.json'
        return osp.join(self.outfile_prefix, filename.format(epoch))

    def evaluate(self, dataset, results, work_dir, epoch, logger=None):

        # preprocess results
        preds = self.process_results(results)

        # build json file
        if self.outfile_prefix is None:
            self.outfile_prefix = osp.join(work_dir, prefix='detections')
        save_file = self.build_file(epoch)

        # convert predictions to coco format and dump to json file
        self.results2json(preds, save_file)

        eval_results = OrderedDict()
        if self.format_only:
            if logger is not None:
                logger.info('results are saved in '
                            f'{osp.dirname(self.outfile_prefix)}')
            return eval_results

        # handle lazy init
        if self._coco_api is None:
            self._coco_api = dataset.coco
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=dataset.metainfo.get('classes'))
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        for metric in self.metrics:
            if logger is not None:
                logger.info(f'Evaluating {metric}...')
            iou_type = 'bbox' if metric == 'proposal' else metric
            try:
                predictions = json.load(open(save_file))
                coco_dt = self._coco_api.loadRes(predictions)
            except IndexError:
                if logger is not None:
                    logger.error(
                        'The testing results of the whole dataset is empty.')
                break
            coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(self.proposal_nums)
            coco_eval.params.iouThrs = self.iou_thrs

            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            metric_items = self.metric_items
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item "{metric_item}" is not supported')

            if metric == 'proposal':
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                if self.classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = coco_eval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, cat_id in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self._coco_api.loadCats(cat_id)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{round(ap, 3)}'))
                        eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    if logger is not None:
                        logger.info('\n' + table.table)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = coco_eval.stats[coco_metric_names[metric_item]]
                    eval_results[key] = float(f'{round(val, 3)}')

                ap = coco_eval.stats[:6]
                if logger is not None:
                    logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                                f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                                f'{ap[4]:.3f} {ap[5]:.3f}')
        return eval_results
