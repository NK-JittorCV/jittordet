import argparse
import os.path as osp

from jittordet.engine import Runner, load_cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
