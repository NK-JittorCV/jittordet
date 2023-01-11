import yaml


def yaml_reader(cfg):
    with open(cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    return cfg


def openmmlab_cfg_reader(config_file):
    raise NotImplementedError


cfg_readers = {
    '.yml': yaml_reader,
    '.ymal': yaml_reader,
    '.py': openmmlab_cfg_reader
}
