import jittor as jt

from ..register import LOOPS
from .base_loop import BaseLoop


@LOOPS.register_module()
class ValLoop(BaseLoop):

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        self._iter = 0

        for idx, data_batch in enumerate(self.runner.val_dataset):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.runner.val_evaluator.evaluate(self.runner.val_dataset,
                                                     self.runner.logger)

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    @jt.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.model(data_batch, phase='predict')
        self.runner.val_evaluator.process(self.runner.val_dataset, outputs)

        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
