# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/modeling/rpn/rpn.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 334 bytes
from FCOS.fcos_core.modeling.rpn.fcos import build_fcos

def build_rpn(cfg, in_channels):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    if cfg.MODEL.FCOS_ON:
        return build_fcos(cfg, in_channels)
# okay decompiling ./fcos_core/modeling/rpn/rpn.pyc
