# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/utils/model_serialization.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 3457 bytes
from collections import OrderedDict
import logging, torch
from FCOS.fcos_core.utils.imports import import_file

def align_and_update_state_dicts(model_state_dict, loaded_state_dict):
    """
    Strategy: suppose that the models that we will create will have prefixes appended
    to each of its keys, for example due to an extra level of nesting that the original
    pre-trained data from ImageNet won't contain. For example, model.state_dict()
    might return backbone[0].body.res2.conv1.weight, while the pre-trained model contains
    res2.conv1.weight. We thus want to match both parameters together.
    For that, we look for each model weight, look among all loaded keys if there is one
    that is a suffix of the current weight name, and use it if that's the case.
    If multiple matches exist, take the one with longest size
    of the corresponding name. For example, for the same model as before, the pretrained
    weight file can contain both res2.conv1.weight, as well as conv1.weight. In this case,
    we want to match backbone[0].body.conv1.weight to conv1.weight, and
    backbone[0].body.res2.conv1.weight to res2.conv1.weight.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))
    match_matrix = [len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(current_keys), len(loaded_keys))
    max_match_size, idxs = match_matrix.max(1)
    idxs[max_match_size == 0] = -1
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = '{: <{}} loaded from {: <{}} of shape {}'
    logger = logging.getLogger(__name__)
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            pass
        else:
            key = current_keys[idx_new]
            key_old = loaded_keys[idx_old]
            model_state_dict[key] = loaded_state_dict[key_old]
            logger.info(log_str_template.format(key, max_size, key_old, max_size_loaded, tuple(loaded_state_dict[key_old].shape)))


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    else:
        stripped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            stripped_state_dict[key.replace(prefix, '')] = value

        return stripped_state_dict


def load_state_dict(model, loaded_state_dict):
    model_state_dict = model.state_dict()
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix='module.')
    align_and_update_state_dicts(model_state_dict, loaded_state_dict)
    model.load_state_dict(model_state_dict)
# okay decompiling ./fcos_core/utils/model_serialization.pyc
