from .base_loop import BaseLoop


class EpochTrainLoop(BaseLoop):

    def __init__(self, runner, max_epoch, val_interval=1):
        super().__init__(runner=runner)
        self.max_epoch = max_epoch
        self.val_interval = val_interval
        self._epoch = 0
        self._iter = 0
        pass

    def run(self):
        self.runner.call_hook('before_train')
        # setup scheduler last_step from -1 to 0
        for scheduler in self.runner.schedulers:
            scheduler.step()

        while self._epoch < self._max_epochs:
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

        for scheduler in self.runner.schedulers:
            if not getattr(scheduler, 'by_iter', False):
                scheduler.step()

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch):
        """Iterate one min-batch."""
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.model(**data_batch, phase='loss')
        self.runner.optimizer.step(outputs['loss'])
        for scheduler in self.runner.schedulers:
            # for warmup scheduler
            if getattr(scheduler, 'by_iter', False):
                scheduler.step()

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
