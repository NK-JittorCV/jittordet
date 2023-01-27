# Modified from OpenMMLab mmdet/models/dense_heads/rpn_head.py
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional
import copy

import jittor as jt
import jittor.nn as nn

from jittordet.engine import MODELS, ConfigDict
from ..layers import ConvModule
from .anchor_head import AnchorHead
from ..utils.nms import batched_nms 

from jittordet.ops.bbox_transforms import scale_boxes
from jittordet.structures import InstanceData

@MODELS.register_module()
class RPNHead(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_convs=1,
                 **kwargs):
        self.num_convs = num_convs
        assert num_classes == 1
        super(RPNHead, self).__init__(
            num_classes,
            in_channels,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        if self.num_convs > 1:
            rpn_convs = []
            for i in range(self.num_convs):
                if i == 0:
                    in_channels = self.in_channels
                else:
                    in_channels = self.feat_channels
                # use ``inplace=False`` to avoid error: one of the variables
                # needed for gradient computation has been modified by an
                # inplace operation.
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        self.feat_channels,
                        3,
                        padding=1))
            self.rpn_conv = nn.Sequential(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(self.in_channels, 
                                      self.feat_channels, 
                                      3, 
                                      padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * self.cls_out_channels,
                                 1)
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_base_priors * reg_dim, 1)

    def execute_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        x = self.rpn_conv(x)
        x = nn.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred
    

    def _predict_by_feat_single(self,
                                cls_score_list: List[jt.Var],
                                bbox_pred_list: List[jt.Var],
                                score_factor_list: List[jt.Var],
                                mlvl_priors: List[jt.Var],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        
        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(zip(cls_score_list, 
                                                                       bbox_pred_list,
                                                                       mlvl_priors)):
            assert cls_score.shape[-2:] == bbox_pred.shape[-2:]
            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0] since mmdet v2.0
                # BG cat_id: 1
                scores = nn.softmax(cls_score ,-1)[:, :-1]
            scores = jt.squeeze(scores, -1)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                rank_inds,ranked_scores = scores.argsort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            level_ids.append(jt.full((scores.size(0), ),level_idx, dtype=jt.int64))
        
        bbox_pred = jt.concat(mlvl_bbox_preds)
        priors = jt.concat(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = jt.concat(mlvl_scores)
        results.level_ids = jt.concat(level_ids)        
        return self._bbox_post_process(results=results, 
                                       cfg=cfg, 
                                       rescale=rescale, 
                                       img_meta=img_meta)

    def _bbox_post_process(self,
                           results: InstanceData,
                           cfg: ConfigDict,
                           rescale: bool = False,
                           with_nms: bool = True,
                           img_meta: Optional[dict] = None) -> InstanceData:
        assert with_nms, '`with_nms` must be True in RPNHead'
        if rescale:
            assert img_meta.get('scale_factor') is not None
            scale_factor = [1 / s for s in img_meta['scale_factor']]
            results.bboxes = scale_boxes(results.bboxes, scale_factor)

        # filter small size bboxes
        if cfg.get('min_bbox_size', -1) >= 0:
            w = results.bboxes[:, 2] - results.bboxes[:, 0]
            h = results.bboxes[:, 3] - results.bboxes[:, 1]
            valid_mask = (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
            if not valid_mask.all():
                results = results[valid_mask]

        if results.bboxes.numel() > 0:
            bboxes = results.bboxes
            # TODO batched_nms
            det_bboxes, _, keep_idxs = batched_nms(bboxes, 
                                                   results.scores,
                                                   results.level_ids, 
                                                   cfg.nms)
            results = results[keep_idxs]
            # some nms would reweight the score, such as softnms
            results.scores = det_bboxes[:, -1]
            results = results[:cfg.max_per_img]
            # TODO: This would unreasonably show the 0th class label
            #  in visualization
            results.labels = jt.zeros(len(results), dtype=jt.int64)
            del results.level_ids
        else:
            # To avoid some potential error
            results_ = InstanceData()
            results_.bboxes = jt.zeros_like(results.bboxes)
            results_.scores = jt.zeros_like(results.scores)
            results_.labels = jt.zeros_like(results.scores)
            results = results_
        return results