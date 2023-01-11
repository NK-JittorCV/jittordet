import os.path as osp
from collections.abc import Mapping

from addict import Dict
from easydict import EasyDict as edict

from .parsers import cfg_parsers
from .readers import cfg_readers

BASE_KEY = '_base_'
COVER_KEY = '_cover_'


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


def load_cfg(cfg):
    """load cfg from different file."""
    assert osp.isfile(cfg), f'{cfg} is not a exist file'
    ext = osp.splitext(cfg)[-1]
    filepath = osp.dirname(cfg)
    if ext not in cfg_readers:
        raise NotImplementedError(f'Cannot parse {cfg} with {ext} type yet')
    reader = cfg_readers[ext]
    cfg = ConfigDict(reader(cfg))
    for parser in cfg_parsers:
        parser(cfg)

    if BASE_KEY in cfg:
        base_cfgs = cfg.pop(BASE_KEY)
        if isinstance(base_cfgs, str):
            base_cfgs = [base_cfgs]
        all_cfg = edict()
        for base_cfg in base_cfgs:
            if base_cfg.startswith('~'):
                base_cfg = osp.expanduser(base_cfg)
            if base_cfg.startswith('.'):
                base_cfg = osp.join(filepath, base_cfg)
            all_cfg.update(load_cfg(base_cfg))
        merge_cfg(all_cfg, cfg)
        cfg = all_cfg

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


def dump_cfg(cfg, save_path):
    """dump cfg into different files."""
    raise NotImplementedError
