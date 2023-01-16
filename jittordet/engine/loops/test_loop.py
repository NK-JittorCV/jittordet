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

        results = []
        for idx, data_batch in enumerate(self.runner.test_dataset):
            results.extend(self.run_iter(idx, data_batch))

        # compute metrics
        evaluator = self.runner.test_evaluator
        metrics = evaluator.evaluate(self.runner.test_dataset, results)
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

    @jt.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.model(**data_batch, phase='predict')
        if jt.in_mpi:
            all_rank_results = []
            for i in range(jt.world_size):
                rank_outputs = [o.mpi_broadcast(root=i) for o in outputs]
                all_rank_results.extend(rank_outputs)
            outputs = all_rank_results

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

        return outputs
