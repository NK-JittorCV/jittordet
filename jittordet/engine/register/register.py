# modified from openmmlab mmengine
# https://github.com/open-mmlab/mmengine/blob/main/mmengine/registry/registry.py

import copy
import inspect
from functools import partial


class Register:

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return key in self._module_dict

    def __repr__(self):
        format_str = self.__class__.__name__ + \
                     f'(name={self._name}, ' \
                     f'items={self._module_dict})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key)

    def _register_module(self, module, module_name=None, force=False):
        if not inspect.isclass(module) and not inspect.isfunction(module):
            raise TypeError('module must be a class or a function, '
                            f'but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(self, name=None, force=False, module=None):
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        # raise the error ahead of time
        if not (name is None or isinstance(name, str)
                or isinstance(name, (list, tuple))):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        # use it as a normal method: x.register_module(module=SomeClass)
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # use it as a decorator: @x.register_module()
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    def build(self, cfg, **default_args):
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg should be a dict, but got {type(cfg)}')

        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError(
                f'default_args should be a dict, but got {type(default_args)}')

        if 'type' not in cfg:
            if default_args is None or 'type' not in default_args:
                raise KeyError(
                    '`cfg` or `default_args` must contain the key "type", '
                    f'but got {cfg}\n{default_args}')

        args = copy.deepcopy(cfg)
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {self.name} registry.')
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if inspect.isclass(obj_cls):
            if hasattr(obj_cls, 'from_cfg'):
                return obj_cls.from_cfg(*args)
            else:
                return obj_cls(**args)
        elif inspect.isfunction(obj_cls):
            return partial(obj_cls, **args)
