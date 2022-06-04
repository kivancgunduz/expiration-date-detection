# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/modeling/detector/generalized_rcnn.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 1556 bytes
"""
Implements the Generalized R-CNN framework
"""
from torch import nn
from FCOS.fcos_core.structures.image_list import to_image_list
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn

class GeneralizedRCNN(nn.Module):
    __doc__ = '\n    Main class for Generalized R-CNN. Currently supports boxes and masks.\n    It consists of three main parts:\n    - backbone\n    - rpn\n    - heads: takes the features + the proposals from the RPN and computes\n        detections / masks from it.\n    '

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        return (proposals, proposal_losses)
# okay decompiling ./fcos_core/modeling/detector/generalized_rcnn.pyc
