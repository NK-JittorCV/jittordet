import datetime
import time
from collections import defaultdict

import jittor as jt
import numpy as np

from ..register import HOOKS
from .base_hook import BaseHook


@HOOKS.register_module()
class LoggerHook(BaseHook):

    def __init__(self,
                 interval=50,
                 interval_exp_name=300,
                 max_log_length=10000):
        self.interval = interval
        self.interval_exp_name = interval_exp_name
        self.max_log_length = max_log_length

        empty_ndarray = lambda: np.zeros([0])  # noqa E731
        self._train_log_history = defaultdict(empty_ndarray)
        self._val_log_history = defaultdict(empty_ndarray)
        self._test_log_history = defaultdict(empty_ndarray)

    def _select_log_history(self, phase):
        assert phase in ['train', 'val', 'test']
        if phase == 'train':
            log_history = self._train_log_history
        elif phase == 'val':
            log_history = self._val_log_history
        else:
            log_history = self._test_log_history
        return log_history

    def update_log_hitory(self, phase, key, value):
        # validate value
        if isinstance(value, jt.Var):
            if jt.in_mpi:
                value = value.mpi_all_reduce('mean')
            value = value.numpy()
        elif not isinstance(value, np.ndarray):
            value = np.array([value], dtype=np.float32)

        # record history
        log_history = self._select_log_history(phase)
        log_value = log_history[key]
        log_value = np.concatenate([log_value, value])
        if log_value.size > self.max_log_length:
            log_value = log_value[-self.max_log_length:]
        log_history[key] = log_value

    def get_log_hitory(self, phase, key, window_size=1, reduction='mean'):
        log_history = self._select_log_history(phase)
        log_value = log_history[key]
        if log_value.size == 0:
            raise RuntimeError(f'{key} has not been logged in history')

        if reduction == 'mean':
            window_size = min(window_size, log_value.size)
            log_value = log_value[-window_size:]
            log_value = log_value.mean().item()
        elif reduction == 'current':
            log_value = log_value[-1].item()
        else:
            raise KeyError('Only support "mean" and "current" reduction, but '
                           f'got {reduction}')
        return log_value

    def format_train_log_str(self, runner, batch_idx):
        log_str_list = []

        # epoch iteration number information
        log_str = '(train) '
        cur_epoch = runner.train_loop.cur_epoch
        max_epoch = runner.train_loop.max_epoch
        iter_per_epoch = len(runner.train_dataset)
        # get the max length of iteration and epoch number
        epoch_len = len(str(max_epoch))
        iter_len = len(str(iter_per_epoch))
        # right just the length
        cur_epoch = str(cur_epoch + 1).rjust(epoch_len)
        cur_iter = str(batch_idx + 1).rjust(iter_len)
        log_str += f'[{cur_epoch}/{max_epoch}]'
        log_str += f'[{cur_iter}/{iter_per_epoch}]'
        log_str_list.append(log_str)

        # iter time and etc time
        iter_time = self.get_log_hitory(
            'train', 'time', 1000, reduction='mean')
        past_iter = runner.train_loop.cur_iter
        total_iter = len(runner.train_dataset) * max_epoch
        eta_time = iter_time * (total_iter - past_iter)
        eta_time = datetime.timedelta(seconds=int(eta_time))
        mm, ss = divmod(eta_time.seconds, 60)
        hh, mm = divmod(mm, 60)
        format_eta_time = f'{eta_time.days:01d} day {hh:02d}:{mm:02d}:{ss:02d}'
        log_str_list.extend(
            [f'eta: {format_eta_time}', f'time: {iter_time:.4f}'])

        # leanring rate information
        lr = runner.optimizer.lr
        log_str_list.append(f'lr: {lr:.4e}')

        # other information
        for key in self._train_log_history.keys():
            if key == 'time':
                continue
            if 'loss' in key or 'acc' in key:
                log_value = self.get_log_hitory(
                    'train', key, self.interval, reduction='mean')
            else:
                log_value = self.get_log_hitory(
                    'train', key, self.interval, reduction='current')
            log_str_list.append(f'{key}: {log_value:.4f}')

        log_str = '  '.join(log_str_list)
        return log_str

    def format_test_val_log_str(self, runner, batch_idx, phase):
        assert phase in ['val', 'test']
        dataset = runner.val_dataset if phase == 'val' else runner.test_dataset
        loop = runner.val_loop if phase == 'val' else runner.test_loop
        log_str_list = []

        # epoch iteration number information
        log_str = '(val) [1/1]' if phase == 'val' else '(test) [1/1]'
        iter_per_epoch = len(dataset)
        # get the max length of iteration and epoch number
        iter_len = len(str(iter_per_epoch))
        # right just the length
        cur_iter = str(batch_idx + 1).rjust(iter_len)
        log_str += f'[{cur_iter}/{iter_per_epoch}]'
        log_str_list.append(log_str)

        # iter time and etc time
        iter_time = self.get_log_hitory(phase, 'time', 1000, reduction='mean')
        past_iter = loop.cur_iter
        eta_time = iter_time * (len(dataset) - past_iter)
        eta_time = datetime.timedelta(seconds=int(eta_time))
        mm, ss = divmod(eta_time.seconds, 60)
        hh, mm = divmod(mm, 60)
        format_eta_time = f'{eta_time.days:01d} day {hh:02d}:{mm:02d}:{ss:02d}'
        log_str_list.extend(
            [f'eta: {format_eta_time}', f'time: {iter_time:.4f}'])

        # other information
        for key in self._select_log_history(phase).keys():
            if key == 'time':
                continue
            log_value = self.get_log_hitory(
                phase, key, self.interval, reduction='current')
            log_str_list.append(f'{key}: {log_value:.4f}')

        log_str = '  '.join(log_str_list)
        return log_str

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.train_iter_start_time = time.time()

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        self.val_iter_start_time = time.time()

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        self.test_iter_start_time = time.time()

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        for key, value in outputs.items():
            self.update_log_hitory('train', key, value)
        iter_time = time.time() - self.train_iter_start_time
        self.update_log_hitory('train', 'time', iter_time)

        if self.every_n_interval(batch_idx, self.interval):
            log_str = self.format_train_log_str(runner, batch_idx)
            runner.logger.info(log_str)

        if self.every_n_interval(batch_idx, self.interval_exp_name):
            runner.logger.info(f'experiment name: {runner.experiment_name}')

    def after_val_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        iter_time = time.time() - self.val_iter_start_time
        self.update_log_hitory('val', 'time', iter_time)

        if self.every_n_interval(batch_idx, self.interval):
            log_str = self.format_test_val_log_str(runner, batch_idx, 'val')
            runner.logger.info(log_str)

        if self.every_n_interval(batch_idx, self.interval_exp_name):
            runner.logger.info(f'experiment name: {runner.experiment_name}')

    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):
        iter_time = time.time() - self.test_iter_start_time
        self.update_log_hitory('test', 'time', iter_time)

        if self.every_n_interval(batch_idx, self.interval):
            log_str = self.format_test_val_log_str(runner, batch_idx, 'test')
            runner.logger.info(log_str)

        if self.every_n_interval(batch_idx, self.interval_exp_name):
            runner.logger.info(f'experiment name: {runner.experiment_name}')

    def after_val_epoch(self, runner, metrics=None):
        log_str = self.format_metrics(metrics)
        if log_str is not None:
            runner.logger.info(log_str)

    def after_test_epoch(self, runner, metrics=None):
        log_str = self.format_metrics(metrics)
        if log_str is not None:
            runner.logger.info(log_str)

    def format_metrics(self, metrics):
        log_str_list = []
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                log_str_list.append(f'{key}: {value:.3f}')
        else:
            log_str_list.append(f'metrics: {metrics}')

        log_str = '  '.join(log_str_list) if log_str_list else None
        return log_str
