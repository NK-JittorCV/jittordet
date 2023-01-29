import os.path as osp

from ..register import HOOKS
from .base_hook import BaseHook


@HOOKS.register_module()
class CheckpointHook(BaseHook):

    def __init__(self, interval=1, by_iter=False):
        assert isinstance(interval, (int, list))
        self.interval = interval
        self.by_iter = by_iter

    def after_train_epoch(self, runner):

        if self.by_iter:
            return None
        cur_epoch = runner.train_loop.cur_epoch

        if isinstance(self.interval, int):
            save_ckpt = self.every_n_interval(cur_epoch, self.interval)
        else:
            save_ckpt = cur_epoch in self.interval

        if save_ckpt:
            ckpt_filepath = osp.join(runner.log_dir,
                                     f'epoch_{cur_epoch + 1}.pkl')
            runner.save_checkpoint(ckpt_filepath)
            runner.logger.info(f'save checkpoint to {ckpt_filepath}')

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        if not self.by_iter:
            return None
        cur_iter = runner.train_loop.cur_iter

        if isinstance(self.interval, int):
            save_ckpt = self.every_n_interval(cur_iter, self.interval)
        else:
            save_ckpt = cur_iter in self.interval

        if save_ckpt:
            ckpt_filepath = osp.join(runner.log_dir,
                                     f'iter_{cur_iter + 1}.pkl')
            runner.save_checkpoint(ckpt_filepath)
            runner.logger.info(f'save checkpoint to {ckpt_filepath}')
