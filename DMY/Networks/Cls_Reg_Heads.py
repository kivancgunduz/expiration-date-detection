# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/DMY/Networks/Cls_Reg_Heads.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 3980 bytes
import torch, math
from torch import nn

class DmYHeads(torch.nn.Module):

    def __init__(self, strides, in_channels, num_convs, num_classes):
        super(DmYHeads, self).__init__()
        self.strides = strides
        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels,
              in_channels,
              kernel_size=3,
              stride=1,
              padding=1,
              bias=True))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(nn.Conv2d(in_channels,
              in_channels,
              kernel_size=3,
              stride=1,
              padding=1,
              bias=True))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', (nn.Sequential)(*cls_tower))
        self.add_module('bbox_tower', (nn.Sequential)(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels,
          num_classes,
          kernel_size=3,
          stride=1,
          padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
          4,
          kernel_size=3,
          stride=1,
          padding=1)
        self.centerness = nn.Conv2d(in_channels,
          1,
          kernel_size=3,
          stride=1,
          padding=1)
        for modules in [self.cls_tower, self.bbox_tower,
         self.cls_logits, self.bbox_pred,
         self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_((l.weight), std=0.01)
                    nn.init.constant_(l.bias, 0)

        bias_value = -math.log(99.0)
        nn.init.constant_(self.cls_logits.bias, bias_value)
        self.conv1x1_1 = nn.Sequential(nn.Conv2d(64, 256,
          kernel_size=1,
          stride=1,
          bias=False), nn.GroupNorm(32, 256), nn.ReLU())
        self.conv1x1_2 = nn.Sequential(nn.Conv2d(256, 256,
          kernel_size=1,
          stride=1,
          bias=False), nn.GroupNorm(32, 256), nn.ReLU())
        self.conv1x1_3 = nn.Sequential(nn.Conv2d(512, 256,
          kernel_size=1,
          stride=1,
          bias=False), nn.GroupNorm(32, 256), nn.ReLU())
        for modules in [self.conv1x1_1, self.conv1x1_2, self.conv1x1_3]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_((l.weight), std=0.01)

    def forward(self, input):
        features = []
        x = input[0]
        x = self.conv1x1_1(x)
        features.append(x)
        x = input[1]
        x = self.conv1x1_2(x)
        features.append(x)
        x = input[2]
        x = self.conv1x1_3(x)
        features.append(x)
        logits = []
        bbox_reg = []
        centerness = []
        for idx, f in enumerate(features):
            cls_tower = self.cls_tower(f)
            box_tower = self.bbox_tower(f)
            logits.append(self.cls_logits(cls_tower))
            centerness.append(self.centerness(box_tower))
            bbox_pred = self.bbox_pred(box_tower).relu()
            bbox_reg.append(bbox_pred * [2, 4, 8][idx])

        return (logits, bbox_reg, centerness, features)
# okay decompiling ./Networks/Cls_Reg_Heads.pyc
