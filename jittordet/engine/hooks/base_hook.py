# copy from OpenMMLab mmengine.hooks
# Copyright (c) OpenMMLab. All rights reserved.


class BaseHook:

    priority = 50

    def before_train(self, runner):
        pass

    def after_train(self, runner):
        pass

    def before_val(self, runner):
        pass

    def after_val(self, runner):
        pass

    def before_test(self, runner):
        pass

    def after_test(self, runner):
        pass

    def before_train_epoch(self, runner):
        self._before_epoch(runner, mode='train')

    def before_val_epoch(self, runner):
        self._before_epoch(runner, mode='val')

    def before_test_epoch(self, runner):
        self._before_epoch(runner, mode='test')

    def after_train_epoch(self, runner):
        self._after_epoch(runner, mode='train')

    def after_val_epoch(self, runner, metrics=None):
        self._after_epoch(runner, mode='val')

    def after_test_epoch(self, runner, metrics=None):
        self._after_epoch(runner, mode='test')

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='train')

    def before_val_iter(self, runner, batch_idx, data_batch=None):
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='val')

    def before_test_iter(self, runner, batch_idx, data_batch=None):
        self._before_iter(
            runner, batch_idx=batch_idx, data_batch=data_batch, mode='test')

    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='train')

    def after_val_iter(self,
                       runner,
                       batch_idx,
                       data_batch=None,
                       outputs=None) -> None:
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='val')

    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None) -> None:
        self._after_iter(
            runner,
            batch_idx=batch_idx,
            data_batch=data_batch,
            outputs=outputs,
            mode='test')

    def _before_epoch(self, runner, mode='train'):
        pass

    def _after_epoch(self, runner, mode='train'):
        pass

    def _before_iter(self,
                     runner,
                     batch_idx,
                     data_batch=None,
                     mode='train') -> None:
        pass

    def _after_iter(self,
                    runner,
                    batch_idx,
                    data_batch=None,
                    outputs=None,
                    mode='train'):
        pass

    def every_n_interval(self, idx, n):
        return (idx + 1) % n == 0 if n > 0 else False
