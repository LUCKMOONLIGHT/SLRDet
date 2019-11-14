import torch
# from mmdet.ops.nms.rotated_overlap import get_iou_matrix
from mmdet.ops.rotated.rotate_nms import rotate_iou as get_iou_matrix
def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    # import time, datetime
    # startTime = time.time()
    assert mode in ['iou', 'iof']
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)
    if is_aligned:
        ious = get_iou_matrix(bboxes1, bboxes2) #(m,1)
        return ious.diagonal()
    else:
        ious = get_iou_matrix(bboxes1, bboxes2) #(m,n)
    return ious

