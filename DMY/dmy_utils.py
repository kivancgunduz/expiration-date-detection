# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/DMY/dmy_utils.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 2553 bytes
import torch
from DMY import cfg

def modify_weights_name(model, DEVICE):
    """
    Modify the corresponding weights names between model_dict and weight_dict.
    """
    model_dict = model.state_dict()
    pretrained_dict = torch.load((cfg.network_config['weight_dmy']), map_location=DEVICE)
    pretrained_dict = {k[7:]:v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)


def load_networks(path):
    """
    Load networks and weights
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(DEVICE))
    cfg.network_config['weight_dmy'] = '{}/{}'.format(path, cfg.network_config['weight_dmy'])
    cfg.network_config['weight_fe'] = '{}/{}'.format(path, cfg.network_config['weight_fe'])
    cfg.network_config['weight_cam'] = '{}/{}'.format(path, cfg.network_config['weight_cam'])
    cfg.network_config['weight_dtd'] = '{}/{}'.format(path, cfg.network_config['weight_dtd'])
    cfg.dataset_config['dict_dir'] = '{}/{}'.format(path, cfg.dataset_config['dict_dir'])
    print('Loading networks...')
    model_dmy = (cfg.network_config['DMY'])(**cfg.network_config['DMY_args'])
    model_fe = (cfg.network_config['FE'])(**cfg.network_config['FE_args'])
    cfg.network_config['CAM_args']['scales'] = model_fe.Iwantshapes()
    model_cam = (cfg.network_config['CAM'])(**cfg.network_config['CAM_args'])
    model_dtd = (cfg.network_config['DTD'])(**cfg.network_config['DTD_args'])
    model_dmy = model_dmy.to(DEVICE)
    model_fe = model_fe.to(DEVICE)
    model_cam = model_cam.to(DEVICE)
    model_dtd = model_dtd.to(DEVICE)
    print('Loading weights...')
    try:
        try:
            model_dmy.load_state_dict(torch.load((cfg.network_config['weight_dmy']), map_location=DEVICE))
        except:
            modify_weights_name(model_dmy, DEVICE)

    finally:
        model_fe.load_state_dict(torch.load((cfg.network_config['weight_fe']), map_location=DEVICE))
        model_cam.load_state_dict(torch.load((cfg.network_config['weight_cam']), map_location=DEVICE))
        model_dtd.load_state_dict(torch.load((cfg.network_config['weight_dtd']), map_location=DEVICE))

    model_dmy.eval()
    model_fe.eval()
    model_cam.eval()
    model_dtd.eval()
    return (
     model_dmy, model_fe, model_cam, model_dtd)
# okay decompiling ./dmy_utils.pyc
