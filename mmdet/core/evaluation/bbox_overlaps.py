import torch
import numpy as np
# from mmdet.ops.nms.rotated_overlap import get_iou_matrix
from mmdet.ops.rotated.rotate_nms import rotate_iou as get_iou_matrix
def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 6)
        bboxes2(ndarray): shape (k, 5)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols))
    if rows * cols == 0:
        return ious
    ious = get_iou_matrix(torch.from_numpy(bboxes1[:, :5]), torch.from_numpy(bboxes2[:, :5])).numpy()
    return ious
