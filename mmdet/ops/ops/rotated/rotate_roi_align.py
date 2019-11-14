# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from mmdet import _Custom as _C

from apex import amp

class _RROIAlign(Function):
    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale, sample_num=0):
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int)
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()
        output = _C.rotate_roi_align_forward(
            features, rois, spatial_scale, out_h, out_w, sample_num
        )

        # return output, argmax  # DEBUG ONLY
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert (feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = feature_size
        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None
        if ctx.needs_input_grad[0]:
            grad_input = _C.rotate_roi_align_backward(
                grad_output.contiguous(),
                rois,
                spatial_scale,
                out_h,
                out_w,
                batch_size,
                num_channels,
                data_height,
                data_width,
                sample_num
            )
        return grad_input, grad_rois, None, None, None


rroi_align = _RROIAlign.apply


class RROIAlign(nn.Module):
    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RROIAlign, self).__init__()
        self.out_size = out_size
        self.spatial_scale = spatial_scale
        self.sample_num = sample_num

    @amp.float_function
    def forward(self, features, rois):
        return rroi_align(
            features, rois, self.out_size, self.spatial_scale, self.sample_num
        )

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}, sample_num={}'.format(
            self.out_size, self.spatial_scale, self.sample_num)
        return format_str
