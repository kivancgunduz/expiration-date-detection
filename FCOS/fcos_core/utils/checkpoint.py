# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/utils/checkpoint.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 4343 bytes
import logging, os, torch
from FCOS.fcos_core.utils.model_serialization import load_state_dict
from FCOS.fcos_core.utils.imports import import_file

class Checkpointer(object):

    def __init__(self, model, optimizer=None, scheduler=None, save_dir='', save_to_disk=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        else:
            if not self.save_to_disk:
                return
            data = {}
            data['model'] = self.model.state_dict()
            if self.optimizer is not None:
                data['optimizer'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                data['scheduler'] = self.scheduler.state_dict()
        data.update(kwargs)
        save_file = os.path.join(self.save_dir, '{}.pth'.format(name))
        self.logger.info('Saving checkpoint to {}'.format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            f = self.get_checkpoint_file()
        if not f:
            self.logger.info('No checkpoint found. Initializing model from scratch')
            return {}
        else:
            self.logger.info('Loading checkpoint from {}'.format(f))
            checkpoint = self._load_file(f)
            self._load_model(checkpoint)
            if 'optimizer' in checkpoint:
                if self.optimizer:
                    self.logger.info('Loading optimizer from {}'.format(f))
                    self.optimizer.load_state_dict(checkpoint.pop('optimizer'))
            if 'scheduler' in checkpoint:
                if self.scheduler:
                    self.logger.info('Loading scheduler from {}'.format(f))
                    self.scheduler.load_state_dict(checkpoint.pop('scheduler'))
            return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        try:
            with open(save_file, 'r') as (f):
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            last_saved = ''

        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, 'last_checkpoint')
        with open(save_file, 'w') as (f):
            f.write(last_filename)

    def _load_file(self, f):
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.load(f, map_location=DEVICE)

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop('model'))


class DetectronCheckpointer(Checkpointer):

    def __init__(self, cfg, model, optimizer=None, scheduler=None, save_dir='', save_to_disk=None, logger=None):
        super(DetectronCheckpointer, self).__init__(model, optimizer, scheduler, save_dir, save_to_disk, logger)
        self.cfg = cfg.clone()

    def _load_file(self, f):
        if f.startswith('catalog://'):
            paths_catalog = import_file('fcos_core.config.paths_catalog', self.cfg.PATHS_CATALOG, True)
            catalog_f = paths_catalog.ModelCatalog.get(f[len('catalog://'):])
            self.logger.info('{} points to {}'.format(f, catalog_f))
            f = catalog_f
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if 'model' not in loaded:
            loaded = dict(model=loaded)
        return loaded
# okay decompiling ./fcos_core/utils/checkpoint.pyc
