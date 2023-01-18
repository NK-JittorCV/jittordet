import warnings
import numpy as np


from typing import List, Optional, Sequence, Union
from collections import OrderedDict

from .base_evaluator import BaseEvaluator
from .utils import eval_map, eval_recalls
from ..register import EVALUATORS


@EVALUATORS.register_module()
class VOCEvaluator(BaseEvaluator):
    def __init__(self, 
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 eval_mode: str = '11points') -> None:
        super().__init__()
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
    
    def process_results(self, results):
        preds = []
        for result in results:
            pred = result['pred_instances']
            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)
            preds.append(dets)
        return preds
    
    def process_dataset(self, dataset, index):
        gts = []
        for i in index:
            gt = dataset.get_data_info(i)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            ann = dict(
                labels=gt_instances['labels'].cpu().numpy(),
                bboxes=gt_instances['bboxes'].cpu().numpy(),
                bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            gts.append(ann)
        return gts
    
    def evaluate(self, 
                 dataset, 
                 results, 
                 work_dir=None, 
                 epoch=None, 
                 logger=None):
        
        # process results
        preds = self.process_results(results)
        # process datasets
        gts = self.process_dataset(dataset, range(len(preds)))
        
        
        eval_results = OrderedDict()
        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_type = dataset.metainfo.get('dataset_type')
            if dataset_type in ['VOC2007', 'VOC2012']:
                dataset_name = 'voc'
                if dataset_type == 'VOC2007' and self.eval_mode != '11points':
                    warnings.warn('Pascal VOC2007 uses `11points` as default '
                                  'evaluate mode, but you are using '
                                  f'{self.eval_mode}.')
                elif dataset_type == 'VOC2012' and self.eval_mode != 'area':
                    warnings.warn('Pascal VOC2012 uses `area` as default '
                                  'evaluate mode, but you are using '
                                  f'{self.eval_mode}.')
            else:
                dataset_name = dataset.metainfo.get('classes')
            mean_aps = []
            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    preds,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    dataset=dataset_name,
                    logger=logger,
                    use_legacy_coordinate=True)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif self.metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in gts]
            recalls = eval_recalls(
                gt_bboxes,
                preds,
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