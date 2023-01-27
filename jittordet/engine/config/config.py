import os
import os.path as osp
from collections.abc import Mapping
from typing import List, Optional, Union

from addict import Dict

from .dumpers import cfg_dumpers
from .parsers import cfg_parsers
from .readers import cfg_readers
from .utils import delete_node

BASE_KEY = '_base_'
COVER_KEY = '_cover_'
RESERVED_KEYS = ['filename']


class ConfigDict(Dict):
    """copy from mmengine https://github.com/open-
    mmlab/mmengine/blob/main/mmengine/config/config.py.

    A dictionary for config which has the same interface as python's built- in
    dictionary and can be used as a normal dictionary. The Config class would
    transform the nested fields (dictionary-like fields) in config file into
    ``ConfigDict``.
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no "
                                 f"attribute '{name}'")
        except Exception as e:
            raise e
        else:
            return value


ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]

MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]


def load_cfg(filepath):
    """load cfg from different file."""
    assert osp.isfile(filepath), f'{filepath} is not a exist file'
    ext = osp.splitext(filepath)[-1]
    if ext not in cfg_readers:
        raise NotImplementedError(
            f'Cannot parse "{filepath}" with {ext} type yet')
    cfg = ConfigDict(cfg_readers[ext](filepath))
    for key in RESERVED_KEYS:
        if key in cfg:
            raise KeyError('f"{key}" is a reserved key')

    # use parsers to translate some leaves
    cfg['filename'] = filepath
    for parser in cfg_parsers:
        parser(cfg)

    if BASE_KEY in cfg:
        base_cfg_paths = cfg.pop(BASE_KEY)
        if isinstance(base_cfg_paths, str):
            base_cfg_paths = [base_cfg_paths]
        all_cfg = ConfigDict()
        root_path = osp.dirname(filepath)
        for base_cfg_path in base_cfg_paths:
            if base_cfg_path.startswith('~'):
                base_cfg_path = osp.expanduser(base_cfg_path)
            if base_cfg_path.startswith('.'):
                base_cfg_path = osp.join(root_path, base_cfg_path)
            all_cfg.update(load_cfg(base_cfg_path))
        merge_cfg(all_cfg, cfg)
        cfg = all_cfg

    delete_node(cfg, COVER_KEY)
    return cfg


def merge_cfg(cfg_a, cfg_b):
    """merge cfg_b into cfg_a."""
    for k, v in cfg_b.items():
        if k in cfg_a and (isinstance(v, Mapping)
                           and isinstance(cfg_a[k], Mapping)):
            if v.pop(COVER_KEY, False):
                cfg_a[k] = v
            else:
                merge_cfg(cfg_a[k], v)
        else:
            cfg_a[k] = v


def dump_cfg(cfg, filepath, allow_exist=False, create_dir=True):
    """dump cfg into different files."""
    ext = osp.splitext(filepath)[-1]
    dir_name = osp.dirname(filepath)
    if ext not in cfg_dumpers:
        raise NotImplementedError(f'Cannot dump cfg to {ext} type file yet')

    if osp.exists(filepath) and not allow_exist:
        raise FileExistsError('The target file has existed')

    if dir_name and not osp.exists(dir_name) and create_dir:
        os.makedirs(dir_name)

    cfg_dumpers[ext](cfg, filepath)
