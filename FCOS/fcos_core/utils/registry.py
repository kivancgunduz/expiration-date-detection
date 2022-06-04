# uncompyle6 version 3.8.0
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.15 (default, Dec 21 2021, 12:03:22) 
# [GCC 10.2.1 20210110]
# Embedded file name: /home/cagatay/PycharmProjects/Expiry/FCOS/fcos_core/utils/registry.py
# Compiled at: 2021-12-16 06:36:40
# Size of source mod 2**32: 1385 bytes


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    __doc__ = '\n    A helper class for managing registering modules, it extends a dictionary\n    and provides a register functions.\n\n    Eg. creeting a registry:\n        some_registry = Registry({"default": default_module})\n\n    There\'re two ways of registering new modules:\n    1): normal way is just calling register function:\n        def foo():\n            ...\n        some_registry.register("foo_module", foo)\n    2): used as decorator when declaring the module:\n        @some_registry.register("foo_module")\n        @some_registry.register("foo_modeul_nickname")\n        def foo():\n            ...\n\n    Access of module is just like using a dictionary, eg:\n        f = some_registry["foo_modeul"]\n    '

    def __init__(self, *args, **kwargs):
        (super(Registry, self).__init__)(*args, **kwargs)

    def register(self, module_name, module=None):
        if module is not None:
            _register_generic(self, module_name, module)
            return
        else:

            def register_fn(fn):
                _register_generic(self, module_name, fn)
                return fn

            return register_fn
# okay decompiling ./fcos_core/utils/registry.pyc
