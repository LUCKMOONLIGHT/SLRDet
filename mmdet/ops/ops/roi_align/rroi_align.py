import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import rroi_align_cuda as _C


class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=2):
        ctx.save_for_backward(rois)
        ctx.out_size = out_size
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.input_shape = features.size()
        output = _C.roi_align_rotated_forward(
            features, rois, spatial_scale, out_size, out_size, sample_num
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        out_size = ctx.out_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            out_size,
            out_size,
            bs,
            ch,
            h,
            w,
            sample_num,
        )
        return grad_input, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply


class ROIAlignRotated(nn.Module):
    def __init__(self, out_size, spatial_scale, sample_num):
        super(ROIAlignRotated, self).__init__()
        self.out_size = out_size
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num

    def forward(self, input, rois):
        return roi_align_rotated(
            input, rois, self.out_size, self.spatial_scale, self.sample_num
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "out_size=" + str(self.out_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sample_num=" + str(self.sample_num)
        tmpstr += ")"
        return tmpstr