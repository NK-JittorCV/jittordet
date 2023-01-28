from typing import List

import jittor as jt


def bbox2roi(bbox_list: List[jt.Var]) -> jt.Var:
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (List[Union[Tensor, :obj:`BaseBoxes`]): a list of bboxes
            corresponding to a batch of images.

    Returns:
        Tensor: shape (n, box_dim + 1), where ``box_dim`` depends on the
        different box types. For example, If the box type in ``bbox_list``
        is HorizontalBoxes, the output shape is (n, 5). Each row of data
        indicates [batch_ind, x1, y1, x2, y2].
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        img_inds = jt.full((bboxes.size(0), 1), img_id, dtype=bboxes.dtype)
        rois = jt.concat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)
    rois = jt.concat(rois_list, 0)
    return rois
