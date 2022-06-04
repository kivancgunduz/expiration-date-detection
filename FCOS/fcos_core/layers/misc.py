# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/layers/misc.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 3984 bytes
"""
helper class that supports empty tensors on some nn functions.

Ideally, add support directly in PyTorch to empty tensors in
those functions.

This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""
import math, torch
from torch.nn.modules.utils import _ntuple

class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return (_NewEmptyTensorOp.apply(grad, shape), None)


class Conv2d(torch.nn.Conv2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        else:
            output_shape = [(i + 2 * p - (di * (k - 1) + 1)) // d + 1 for i, p, di, k, d in zip(x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride)]
            output_shape = [
             x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)


class BatchNorm2d(torch.nn.BatchNorm2d):

    def forward(self, x):
        if x.numel() > 0:
            return super(BatchNorm2d, self).forward(x)
        else:
            output_shape = x.shape
            return _NewEmptyTensorOp.apply(x, output_shape)


class DFConv2d(torch.nn.Module):
    __doc__ = 'Deformable convolutional layer'

    def __init__(self, in_channels, out_channels, with_modulated_dcn=True, kernel_size=3, stride=1, groups=1, padding=1, dilation=1, deformable_groups=1, bias=False):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            from fcos_core.layers import ModulatedDeformConv
            offset_channels = offset_base_channels * 3
            conv_block = ModulatedDeformConv
        else:
            from fcos_core.layers import DeformConv
            offset_channels = offset_base_channels * 2
            conv_block = DeformConv
        self.offset = Conv2d(in_channels,
          (deformable_groups * offset_channels),
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          groups=1,
          dilation=dilation)
        for l in [self.offset]:
            torch.nn.init.kaiming_uniform_((l.weight), a=1)
            torch.nn.init.constant_(l.bias, 0.0)

        self.conv = conv_block(in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          dilation=dilation,
          groups=groups,
          deformable_groups=deformable_groups,
          bias=bias)
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.offset_base_channels = offset_base_channels

    def forward(self, x):
        assert x.numel() > 0, 'only non-empty tensors are supported'
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                split_point = self.offset_base_channels * 2
                offset = offset_mask[:, :split_point, :, :]
                mask = offset_mask[:, split_point:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x
# okay decompiling ./fcos_core/layers/misc.pyc
