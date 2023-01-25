import jittor as jt

from ..register import LOOPS
from .base_loop import BaseLoop


@LOOPS.register_module()
class TestLoop(BaseLoop):

    def run(self):
        """Launch validation."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        self._iter = 0

        for idx, data_batch in enumerate(self.runner.test_dataset):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.runner.test_evaluator.evaluate(self.runner.test_dataset,
                                                      self.runner.logger)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

    @jt.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.model(data_batch, phase='predict')
        self.runner.test_evaluator.process(self.runner.test_dataset, outputs)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
