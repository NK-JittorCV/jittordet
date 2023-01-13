import os
import os.path as osp
import re

from .utils import iter_leaves, set_leaf


def env_variable_parsers(cfg):
    """use environment variables in cfg."""
    regexp1 = r'^\s*\$(\w+)\s*\:\s*(\S*?)\s*$'
    regexp2 = r'\{\s*\$(\w+)\s*\:\s*(\S*?)\s*\}'
    for keys, value in iter_leaves(cfg):
        if not isinstance(value, str):
            continue
        # entire string is environment
        results = re.match(regexp1, value)
        if results:
            var_name, def_value = results.groups()
            new_value = os.environ[var_name] \
                if var_name in os.environ else def_value
            if new_value.isdigit():
                new_value = eval(new_value)
            set_leaf(cfg, keys, new_value)
            continue
        # partial string is environment
        results = re.findall(regexp2, value)
        for var_name, def_value in results:
            regexp = r'\{\s*\$' + var_name + r'\s*\:\s*' \
                + def_value + r'\s*\}'
            new_value = os.environ[var_name] if var_name in os.environ \
                else def_value
            value = re.sub(regexp, new_value, value)
        set_leaf(cfg, keys, value)


def default_var_parsers(cfg):
    """set some default value in cfg."""
    filename = cfg['filename']
    file_dirname = osp.dirname(filename)
    file_basename = osp.basename(filename)
    file_basename_no_extension = osp.splitext(file_basename)[0]
    file_extname = osp.splitext(filename)[1]
    support_templates = dict(
        fileDirname=file_dirname,
        fileBasename=file_basename,
        fileBasenameNoExtension=file_basename_no_extension,
        fileExtname=file_extname)

    for keys, value in iter_leaves(cfg):
        if not isinstance(value, str):
            continue

        for k, v in support_templates.items():
            regexp = r'\{\s*' + str(k) + r'\s*\}'
            v = v.replace('\\', '/')
            value = re.sub(regexp, v, value)

        set_leaf(cfg, keys, value)


cfg_parsers = [
    env_variable_parsers,
    default_var_parsers,
]
