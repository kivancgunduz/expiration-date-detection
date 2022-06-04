# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/layers/scale.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 270 bytes
import torch
from torch import nn

class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale
# okay decompiling ./fcos_core/layers/scale.pyc
