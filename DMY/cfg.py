# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/DMY/cfg.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 985 bytes
from DMY.Networks.DmY_Network import Network
from DAN.Networks.DAN import *
network_config = {'DMY':Network, 
 'DMY_args':{'strides':[
   2, 1, 2, 1, 2], 
  'input_shape':[
   3, 64, 256], 
  'in_channels':256, 
  'num_convs':4, 
  'num_classes':3}, 
 'FE':Feature_Extractor, 
 'FE_args':{'strides':[
   (1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)], 
  'compress_layer':False, 
  'input_shape':[
   1, 32, 128]}, 
 'CAM':CAM, 
 'CAM_args':{'maxT':25, 
  'depth':8, 
  'num_channels':64}, 
 'DTD':DTD, 
 'DTD_args':{'nclass':38, 
  'nchannel':512, 
  'dropout':0.3}, 
 'weight_dmy':'data/model_dmy.pth', 
 'weight_fe':'data/dan_model_1.pth', 
 'weight_cam':'data/dan_model_2.pth', 
 'weight_dtd':'data/dan_model_3.pth'}
dataset_config = {'dict_dir':'data/dic_36.txt', 
 'case_sensitive':True}
# okay decompiling ./cfg.pyc
