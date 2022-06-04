# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/modeling/make_layers.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 3476 bytes
"""
Miscellaneous utility functions
"""
import torch
from torch import nn
from FCOS.fcos_core.config import cfg
from FCOS.fcos_core.layers import Conv2d

def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    if not dim_per_gp == -1:
        assert num_groups == -1, 'GroupNorm: can only specify G or C/G.'
        assert dim_per_gp > 0 and dim % dim_per_gp == 0, 'dim: {}, dim_per_gp: {}'.format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, 'dim: {}, num_groups: {}'.format(dim, num_groups)
        group_gn = num_groups
    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON
    return torch.nn.GroupNorm(get_group_gn(out_channels, dim_per_gp, num_groups), out_channels, eps, affine)


def make_conv3x3(in_channels, out_channels, dilation=1, stride=1, use_gn=False, use_relu=False, kaiming_init=True):
    conv = Conv2d(in_channels,
      out_channels,
      kernel_size=3,
      stride=stride,
      padding=dilation,
      dilation=dilation,
      bias=(False if use_gn else True))
    if kaiming_init:
        nn.init.kaiming_normal_((conv.weight),
          mode='fan_out', nonlinearity='relu')
    else:
        torch.nn.init.normal_((conv.weight), std=0.01)
    if not use_gn:
        nn.init.constant_(conv.bias, 0)
    module = [
     conv]
    if use_gn:
        module.append(group_norm(out_channels))
    if use_relu:
        module.append(nn.ReLU(inplace=True))
    if len(module) > 1:
        return (nn.Sequential)(*module)
    else:
        return conv


def make_fc(dim_in, hidden_dim, use_gn=False):
    """
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    """
    if use_gn:
        fc = nn.Linear(dim_in, hidden_dim, bias=False)
        nn.init.kaiming_uniform_((fc.weight), a=1)
        return nn.Sequential(fc, group_norm(hidden_dim))
    else:
        fc = nn.Linear(dim_in, hidden_dim)
        nn.init.kaiming_uniform_((fc.weight), a=1)
        nn.init.constant_(fc.bias, 0)
        return fc


def conv_with_kaiming_uniform(use_gn=False, use_relu=False):

    def make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1):
        conv = Conv2d(in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=(dilation * (kernel_size - 1) // 2),
          dilation=dilation,
          bias=(False if use_gn else True))
        nn.init.kaiming_uniform_((conv.weight), a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [
         conv]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return (nn.Sequential)(*module)
        else:
            return conv

    return make_conv
# okay decompiling ./fcos_core/modeling/make_layers.pyc
