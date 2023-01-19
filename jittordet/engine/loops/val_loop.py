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

        results = []
        for idx, data_batch in enumerate(self.runner.val_dataset):
            results.extend(self.run_iter(idx, data_batch))
        if len(results) > self.runner.val_dataset.total_len:
            results = results[:self.runner.val_dataset.total_len]

        # compute metrics
        evaluator = self.runner.val_evaluator
        metrics = evaluator.evaluate(self.runner.val_dataset, results)
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')

    @jt.no_grad()
    def run_iter(self, idx, data_batch):
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)

        outputs = self.runner.model(**data_batch, phase='predict')
        if jt.in_mpi:
            all_rank_results = []
            for i in range(jt.world_size):
                rank_outputs = [o.mpi_broadcast(root=i) for o in outputs]
                all_rank_results.extend(rank_outputs)
            outputs = all_rank_results

        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)

        return outputs
