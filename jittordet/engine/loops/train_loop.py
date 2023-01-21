from ..register import LOOPS
from .base_loop import BaseLoop


@LOOPS.register_module()
class EpochTrainLoop(BaseLoop):

    def __init__(self, runner, max_epoch, val_interval=1):
        super().__init__(runner=runner)
        self.val_interval = val_interval
        self._max_epoch = max_epoch

    def run(self):
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epoch:
            self.run_epoch()

            if self._epoch % self.val_interval == 0:
                if self.runner.val_loop is not None:
                    self.runner.val_loop.run()

        self.runner.call_hook('after_train')

    def run_epoch(self):
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')

        self.runner.model.train()
        for idx, data_batch in enumerate(self.runner.train_dataset):
            self.run_iter(idx, data_batch)

        for _scheduler in self.runner.scheduler:
            if not getattr(_scheduler, 'by_iter', False):
                _scheduler.step()

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch):
        """Iterate one min-batch."""
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)

        loss, loss_vars = self.runner.model(data_batch, phase='loss')
        self.runner.optimizer.step(loss)
        for _scheduler in self.runner.scheduler:
            # for warmup scheduler
            if getattr(_scheduler, 'by_iter', False):
                _scheduler.step()

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=loss_vars)
        self._iter += 1
