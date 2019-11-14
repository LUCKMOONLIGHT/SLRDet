import numpy as np
import torch

# from . import nms_cpu, nms_cuda
# from .soft_nms_cpu import soft_nms_cpu
# from .rotated_nms import rotate_cpu_nms
from mmdet.ops.rotated.rotate_nms import RotateNMS, RotateSoftNMS
def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    nms_rot = RotateNMS(iou_thr)
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_tensor = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_tensor = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))
    if dets_tensor.shape[0] == 0:
        keep = dets_tensor.new_zeros(0, dtype=torch.long)
    else:
        keep = nms_rot(dets_tensor[:, :5], dets_tensor[:, 5]).cpu().numpy()
    if is_numpy:
        keep = keep.cpu().numpy()
    return dets[keep, :], keep


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    rotate_nms_instance = RotateSoftNMS(nms_thresh=iou_thr,
                                        sigma=sigma, score_thresh=min_score, method=method)

    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_tensor = dets.detach().cpu()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_tensor = torch.from_numpy(dets)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    inds = rotate_nms_instance(
        dets_tensor[:, :5], dets_tensor[:, 5])
    new_dets = dets_tensor[inds, :]

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)
