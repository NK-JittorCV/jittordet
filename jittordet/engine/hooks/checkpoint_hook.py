import os.path as osp

import jittor as jt

from ..register import HOOKS
from .base_hook import BaseHook


@HOOKS.register_module()
class CheckpointHook(BaseHook):

    def __init__(self, interval=1):
        assert isinstance(interval, (int, list))
        self.interval = interval

    def after_train_epoch(self, runner):
        cur_epoch = runner.train_loop.cur_epoch
        if isinstance(self.interval, int):
            save_ckpt = self.every_n_interval(cur_epoch, self.interval)
        else:
            save_ckpt = cur_epoch in self.interval

        if save_ckpt:
            ckpt_filepath = osp.join(runner.log_dir,
                                     f'ckpt_{cur_epoch + 1}.pkl')
            runner.save_checkpoint(ckpt_filepath)

            if jt.rank == 0:
                runner.logger.info(f'save checkpoint to {ckpt_filepath}')
