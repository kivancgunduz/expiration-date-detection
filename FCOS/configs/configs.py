# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22)
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/configs/configs.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 1753 bytes
import argparse, torch
from FCOS.fcos_core.config import cfg

def fcos_cfgs(path):
    parser = argparse.ArgumentParser(description='Expiration Date Detection Demo')
    parser.add_argument('--config-file',
      default=('{}/data/configs.yaml'.format(path)),
      metavar='FILE',
      help='path to config file')
    parser.add_argument('--weights',
      default=('{}/data/model_expdate.pth'.format(path)),
      metavar='FILE',
      help='path to the trained model')
    parser.add_argument('--images-dir',
      default=('{}/images/'.format(path)),
      metavar='DIR',
      help='path to demo images directory')
    parser.add_argument('--min-image-size',
      type=int,
      default=800,
      help='Smallest size of the image to feed to the model. Model was trained with 800, which gives best results')
    parser.add_argument('opts',
      help='Modify model config options using the command-line',
      default=None,
      nargs=(argparse.REMAINDER))
    parser.add_argument('--thresholds_for_classes',
      type=int,
      default=[
     0.4574, 0.4018, 0.5471, 0.6423],
      help='thresholds values for each classes in the order of non-exp, exp, date, prod')
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.freeze()
    return (
     args, cfg)
# okay decompiling ./configs/configs.pyc
