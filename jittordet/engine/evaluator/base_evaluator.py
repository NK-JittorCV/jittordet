import os
import os.path as osp
import pickle
import shutil
import tempfile
from abc import ABCMeta, abstractmethod

import jittor as jt


class BaseEvaluator(metaclass=ABCMeta):

    @abstractmethod
    def process(self, dataset, data_samples):
        pass

    @abstractmethod
    def compute_metrics(self, dataset, results, logger):
        pass

    def evaluate(self, dataset, logger):
        results = self.collect_results(self.results, len(dataset.data_list))
        if not jt.in_mpi or jt.rank == 0:
            metrics = self.compute_metrics(dataset, results, logger)
        else:
            metrics = None
        metrics = self.broadcast_metrics(metrics)
        self.results.clear()
        return metrics

    def gen_broadcasted_tmpdir(self):
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = jt.full((MAX_LEN, ), 32, dtype=jt.uint8)
        if jt.rank == 0:
            if not osp.exists('.dist_test'):
                os.makedirs('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = jt.array(bytearray(tmpdir.encode()), dtype=jt.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        dir_tensor = dir_tensor.mpi_broadcast(root=0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
        return tmpdir

    def collect_results(self, result_part, size):
        """Collect results under cpu mode."""
        rank, world_size = jt.rank, jt.world_size
        if world_size == 1:
            return result_part[:size]

        tmpdir = self.gen_broadcasted_tmpdir()
        # dump the part result to the dir
        with open(osp.join(tmpdir, f'part_{rank}.pkl'),
                  'wb') as f:  # type: ignore
            pickle.dump(result_part, f, protocol=2)

        self.barrier()

        if rank != 0:
            return None
        else:
            # load results of all parts from tmp dir
            part_list = []
            for i in range(world_size):
                path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
                with open(path, 'rb') as f:
                    part_list.append(pickle.load(f))
            # sort the results
            ordered_results = []
            for res in zip(*part_list):
                ordered_results.extend(list(res))
            # the dataloader may pad some samples
            ordered_results = ordered_results[:size]
            # remove tmp dir
            shutil.rmtree(tmpdir)  # type: ignore
            return ordered_results

    def broadcast_metrics(self, metrics):
        if jt.world_size == 1:
            return metrics

        tmpdir = self.gen_broadcasted_tmpdir()
        if jt.rank == 0:
            with open(osp.join(tmpdir, 'metrics.pkl'), 'wb') as f:
                pickle.dump(metrics, f, protocol=2)

        self.barrier()

        with open(osp.join(tmpdir, 'metrics.pkl'), 'rb') as f:
            metrics = pickle.load(f)

        if jt.rank == 0:
            shutil.rmtree(tmpdir)  # type: ignore
        return metrics

    @staticmethod
    def barrier():
        if jt.in_mpi:
            lock = jt.array([1])
            lock = lock.mpi_all_reduce('mean')
            lock.sync(device_sync=True)
            del lock
