# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/DMY/Networks/DmY_Network.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 604 bytes
import torch.nn as nn
from DMY.Backbone.resnet import resnet45
from DMY.Networks.Cls_Reg_Heads import DmYHeads

class Network(nn.Module):

    def __init__(self, strides, input_shape, in_channels, num_convs, num_classes):
        super(Network, self).__init__()
        self.backbone = resnet45(strides)
        self.det_heads = DmYHeads(strides, in_channels, num_convs, num_classes)

    def forward(self, imgs):
        features = self.backbone(imgs)
        cls_pred, bbox_pred, centerness, final_feature = self.det_heads(features)
        return (
         cls_pred, bbox_pred, centerness, final_feature)
# okay decompiling ./Networks/DmY_Network.pyc
