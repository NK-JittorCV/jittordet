import ast
import json
import os.path as osp
import sys
import types
from importlib import import_module

import yaml


def yaml_reader(filepath):
    with open(filepath, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg


def json_reader(filepath):
    with open(filepath, 'r') as f:
        cfg = json.load(f)
    return cfg


def python_reader(filepath):
    """Reader python type config.

    Refer to mmcv.utils.config.
    """
    # validate python syntax
    with open(filepath, 'r', encoding='utf-8') as f:
        # Setting encoding explicitly to resolve coding issue on windows
        content = f.read()
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise SyntaxError('There are syntax errors in config '
                          f'file {filepath}: {e}')

    filepath = osp.splitext(filepath)[0]
    dir_name = osp.dirname(filepath)
    module_name = osp.basename(filepath)
    sys.path.insert(0, dir_name)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg = {
        name: value
        for name, value in mod.__dict__.items() if not name.startswith('__')
        and not isinstance(value, types.ModuleType)
        and not isinstance(value, types.FunctionType)
    }
    # delete imported module
    del sys.modules[module_name]
    return cfg


cfg_readers = {
    '.yml': yaml_reader,
    '.ymal': yaml_reader,
    '.json': json_reader,
    '.py': python_reader
}
