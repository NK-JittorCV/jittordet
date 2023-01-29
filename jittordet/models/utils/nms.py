# Modified from OpenMMLab mmcv/ops/nms.py
from typing import Dict, Optional, Tuple

import jittor as jt

from jittordet.engine import ConfigType


def batched_nms(boxes: jt.Var,
                scores: jt.Var,
                idxs: jt.Var,
                nms_cfg: Optional[Dict],
                class_agnostic: bool = False) -> Tuple[jt.Var, jt.Var]:
    r"""Performs non-maximum suppression in a batched fashion.

    Modified from `torchvision/ops/boxes.py#L39
    <https://github.com/pytorch/vision/blob/
    505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39>`_.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Note:
        In v1.4.1 and later, ``batched_nms`` supports skipping the NMS and
        returns sorted raw results when `nms_cfg` is None.

    Args:
        boxes (torch.Tensor): boxes in shape (N, 4) or (N, 5).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict | optional): Supports skipping the nms when `nms_cfg`
            is None, otherwise it should specify nms type and other
            parameters like `iou_thr`. Possible keys includes the following.

            - iou_threshold (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
              number of boxes is large (e.g., 200k). To avoid OOM during
              training, the users could set `split_thr` to a small value.
              If the number of boxes is greater than the threshold, it will
              perform NMS on each group of boxes separately and sequentially.
              Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class. Defaults to False.

    Returns:
        tuple: kept dets and indice.

        - boxes (Tensor): Bboxes with score after nms, has shape
          (num_bboxes, 5). last dimension 5 arrange as
          (x1, y1, x2, y2, score)
        - keep (Tensor): The indices of remaining boxes in input
          boxes.
    """
    # skip nms when nms_cfg is None
    if nms_cfg is None:
        scores, inds = scores.sort(descending=True)
        boxes = boxes[inds]
        return boxes, scores, inds

    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        # When using rotated boxes, only apply offsets on center.
        if boxes.size(-1) == 5:
            # Strictly, the maximum coordinates of the rotating box
            # (x,y,w,h,a) should be calculated by polygon coordinates.
            # But the conversion from rotated box to polygon will
            # slow down the speed.
            # So we use max(x,y) + max(w,h) as max coordinate
            # which is larger than polygon max coordinate
            # max(x1, y1, x2, y2,x3, y3, x4, y4)
            max_coordinate = boxes[..., :2].max() + boxes[..., 2:4].max()
            offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
            boxes_ctr_for_nms = boxes[..., :2] + offsets[:, None]
            boxes_for_nms = jt.concat([boxes_ctr_for_nms, boxes[..., 2:5]],
                                      dim=-1)
        else:
            max_coordinate = boxes.max()
            offsets = idxs.astype(boxes.dtype) * (max_coordinate + 1)
            boxes_for_nms = boxes + offsets[:, None]

    # only support jt.nms for now
    nms_cfg_.pop('type', 'nms')
    dets = jt.concat([boxes_for_nms, scores[:, None]], dim=1)
    keep = jt.nms(dets, **nms_cfg_)
    boxes = boxes[keep]
    scores = scores[keep]
    return boxes, scores, keep


def multiclass_nms(multi_bboxes: jt.Var,
                   multi_scores: jt.Var,
                   score_thr: float,
                   nms_cfg: ConfigType,
                   max_num: int = -1,
                   score_factors: Optional[jt.Var] = None,
                   return_inds: bool = False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (Union[:obj:`ConfigDict`, dict]): a dict that contains
            the arguments of nms operations.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.
        box_dim (int): The dimension of boxes. Defaults to 4.

    Returns:
        Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
            (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    labels = jt.arange(num_classes, dtype=jt.int64)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    # NonZero not supported  in TensorRT
    inds = valid_mask.nonzero().squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        if return_inds:
            return bboxes, scores, labels, inds
        else:
            return bboxes, scores, labels

    bboxes, scores, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        bboxes = bboxes[:max_num]
        scores = scores[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return bboxes, scores, labels[keep], inds[keep]
    else:
        return bboxes, scores, labels[keep]
