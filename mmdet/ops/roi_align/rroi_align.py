import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import rroi_align_cuda as _C


class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, out_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.out_size = out_size
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_rotated_forward(
            input, roi, spatial_scale, out_size, out_size, sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        out_size = ctx.out_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
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
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply


class ROIAlignRotated(nn.Module):
    def __init__(self, out_size, spatial_scale, sampling_ratio):
        super(ROIAlignRotated, self).__init__()
        self.out_size = out_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        return roi_align_rotated(
            input, rois, self.out_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "out_size=" + str(self.out_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr