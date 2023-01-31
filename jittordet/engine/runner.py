import copy
import inspect
import os
import os.path as osp
import random
import time

import jittor as jt
import jittor.nn as nn
import numpy as np
from jittor.dataset import Dataset
from jittor.optim import Optimizer

from .config import dump_cfg
from .evaluator import BaseEvaluator
from .hooks import BaseHook
from .logger import get_logger
from .loops import BaseLoop
from .register import (DATASETS, EVALUATORS, HOOKS, LOOPS, MODELS, OPTIMIZERS,
                       SCHEDULERS)

__all__ = ['Runner']


class Runner:

    def __init__(self,
                 model,
                 work_dir,
                 train_dataset=None,
                 val_dataset=None,
                 test_dataset=None,
                 train_loop=None,
                 val_loop=None,
                 test_loop=None,
                 val_evaluator=None,
                 test_evaluator=None,
                 optimizer=None,
                 scheduler=None,
                 hooks=None,
                 seed=None,
                 disable_cuda=False,
                 resume_from=None,
                 load_from=None,
                 experiment_name=None,
                 cfg=None):
        # setup work dir
        self._work_dir = osp.abspath(work_dir)
        if not osp.exists(self._work_dir) and jt.rank == 0:
            os.makedirs(self._work_dir)

        # hold on original configs
        self.cfg = cfg
        if self.cfg is not None:
            self.cfg = copy.deepcopy(self.cfg)

        # set random seed and other randomness related setting
        self.setup_randomness(seed)

        timestamp = jt.array(int(time.time()), dtype=jt.int64)
        if jt.in_mpi:
            timestamp = timestamp.mpi_broadcast(root=0)
        self._time_stamp = time.strftime('%Y%m%d_%H%M%S',
                                         time.localtime(timestamp.item()))

        # setup experiment name
        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._time_stamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._time_stamp}'
        else:
            self._experiment_name = self._time_stamp

        # setup hooks
        self.setup_hooks(hooks)

        self.disable_cuda = disable_cuda
        self._load_from = load_from
        self._resume_from = resume_from
        self.logger = None

        # build model
        self.model = self.build_model(model)

        # hold configs related to the training for lazy initialization
        self.train_dataset = train_dataset
        self.train_loop = train_loop
        self.optimizer = optimizer
        self.scheduler = scheduler

        # hold configs related to the val for lazy initialization
        self.val_dataset = val_dataset
        self.val_loop = val_loop
        self.val_evaluator = val_evaluator

        # hold configs related to the test for lazy initialization
        self.test_dataset = test_dataset
        self.test_loop = test_loop
        self.test_evaluator = test_evaluator

    @classmethod
    def from_cfg(cls, cfg):
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataset=cfg.get('train_dataset'),
            val_dataset=cfg.get('val_dataset'),
            test_dataset=cfg.get('test_dataset'),
            train_loop=cfg.get('train_loop'),
            val_loop=cfg.get('val_loop'),
            test_loop=cfg.get('test_loop'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            optimizer=cfg.get('optimizer'),
            scheduler=cfg.get('scheduler'),
            hooks=cfg.get('hooks'),
            seed=cfg.get('seed'),
            disable_cuda=cfg.get('disable_cuda', False),
            resume_from=cfg.get('resume_from'),
            load_from=cfg.get('load_from'),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg)
        return runner

    @property
    def work_dir(self):
        return self._work_dir

    @property
    def time_stamp(self):
        return self._time_stamp

    @property
    def experiment_name(self):
        return self._experiment_name

    def init_model_weights(self):
        self.logger.info('Start initialize model weights.')

        def dfs_run_init_weights(model):
            for m in model._modules.values():
                dfs_run_init_weights(m)

            if hasattr(model, 'init_weights'):
                model.init_weights()

        dfs_run_init_weights(self.model)
        self.logger.info('Initialize succeed.')

    def build_model(self, model):
        if isinstance(model, nn.Module):
            return model
        elif isinstance(model, dict):
            return MODELS.build(model)
        else:
            raise TypeError('model should be a nn.Module object or dict')

    def build_dataset(self, dataset):
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, dict):

            return DATASETS.build(dataset)
        else:
            raise TypeError(
                'dataset should be a jittor.dataset.Dataset object or dict')

    def build_optimizer(self, optimizer):
        if isinstance(optimizer, Optimizer):
            return Optimizer

        assert isinstance(optimizer, dict)
        optimizer = copy.deepcopy(optimizer)
        cfg_clip_grad = optimizer.pop('clip_grad', None)
        optimizer = OPTIMIZERS.build(optimizer, params=self.model.parameters())
        # hold clip grad config
        optimizer._cfg_clip_grad = cfg_clip_grad

        return optimizer

    def build_scheduler(self, schedulers, optimizer):
        assert isinstance(optimizer, Optimizer), \
            'optimzer should be built before building scheduler.'

        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        param_schedulers = []
        for scheduler in schedulers:
            # recently, scheduler doesn't have a base class
            if not inspect.isclass(scheduler):
                assert isinstance(scheduler, dict)
                scheduler = SCHEDULERS.build(scheduler, optimizer=optimizer)
            param_schedulers.append(scheduler)

        # setup scheduler last_step from -1 to 0
        for scheduler in param_schedulers:
            scheduler.step()
        return param_schedulers

    def build_loop(self, loop):
        if isinstance(loop, BaseLoop):
            return loop
        elif isinstance(loop, dict):
            return LOOPS.build(loop, runner=self)
        else:
            raise TypeError('loops should be a BaseLoop object or dict')

    def build_evaluator(self, evaluator):
        if isinstance(evaluator, BaseEvaluator):
            return evaluator
        elif isinstance(evaluator, dict):
            return EVALUATORS.build(evaluator)
        else:
            raise TypeError(
                'evaluator should be a BaseEvaluator object or dict')

    def train(self):
        self.setup_logger_dir('train')

        train_related = (self.train_dataset, self.train_loop, self.optimizer,
                         self.scheduler)
        assert all([item is not None for item in train_related]), \
            'In training phase, "train_dataset", "train_loop", "optimizer", ' \
            'and "scheduler" should not be None.'
        self.train_dataset = self.build_dataset(self.train_dataset)
        self.train_loop = self.build_loop(self.train_loop)
        self.optimizer = self.build_optimizer(self.optimizer)
        self.scheduler = self.build_scheduler(self.scheduler, self.optimizer)

        val_related = (self.val_dataset, self.val_loop, self.val_evaluator)
        if any([item is not None for item in val_related]):
            assert all([item is not None for item in val_related]), \
                'In training phase, "val_dataset", "val_loop", ' \
                'and "val_evaluator" should be all None or not None.'
            self.val_dataset = self.build_dataset(self.val_dataset)
            self.val_loop = self.build_loop(self.val_loop)
            self.val_evaluator = self.build_evaluator(self.val_evaluator)

        # initialization
        if not self.load_or_resume():
            self.init_model_weights()
        if not self.disable_cuda:
            jt.flags.use_cuda = 1

        # start run
        self.train_loop.run()

    def val(self):
        self.setup_logger_dir('val')

        val_related = (self.val_dataset, self.val_loop, self.val_evaluator)
        assert all([item is not None for item in val_related]), \
            'In val phase, "val_dataset", "val_loop", and "val_evaluator" ' \
            'should be all None or not None.'
        self.val_dataset = self.build_dataset(self.val_dataset)
        self.val_loop = self.build_loop(self.val_loop)
        self.val_evaluator = self.build_evaluator(self.val_evaluator)

        # initialization
        if not self.load_or_resume():
            raise RuntimeError('Model has not been load in val')
        if not self.disable_cuda:
            jt.flags.use_cuda = 1

        # start run
        self.val_loop.run()

    def test(self):
        self.setup_logger_dir('test')

        test_related = (self.test_dataset, self.test_loop, self.test_evaluator)
        assert all([item is not None for item in test_related]), \
            'In test phase, "test_dataset", "test_loop", and ' \
            '"test_evaluator" should be all None or not None.'
        self.test_dataset = self.build_dataset(self.test_dataset)
        self.test_loop = self.build_loop(self.test_loop)
        self.test_evaluator = self.build_evaluator(self.test_evaluator)

        # initialization
        if not self.load_or_resume():
            raise RuntimeError('Model has not been load in test')
        if not self.disable_cuda:
            jt.flags.use_cuda = 1

        # start run
        self.test_loop.run()

    def setup_logger_dir(self, phase):
        assert phase in ['train', 'val', 'test']
        log_dir = osp.join(self.work_dir, self.time_stamp + '_' + phase)
        log_file = osp.join(log_dir, self.time_stamp + '.log')
        if jt.rank == 0 and not osp.exists(log_dir):
            os.makedirs(log_dir)
        self.log_dir = log_dir
        self.logger = get_logger(name='jittordet', log_file=log_file)

        cfg_filename = self.cfg.get('filename', None)
        cfg_filename = 'config.yml' if cfg_filename is None else \
            osp.basename(cfg_filename)
        if jt.rank == 0:
            dump_cfg(self.cfg, osp.join(self.log_dir, cfg_filename))

    def setup_randomness(self, seed):
        if seed is not None:
            rank_diff_seed = seed + jt.rank
            random.seed(rank_diff_seed)
            np.random.seed(rank_diff_seed)
            jt.set_global_seed(seed)

    def load_or_resume(self):
        assert not (self._load_from and self._resume_from), \
            'Can only set either "load_from" or "resume_from"'
        if self._load_from:
            self.load(self._load_from)
            return True
        if self._resume_from:
            self.resume(self._resume_from)
            return True
        return False

    def resume(self, checkpoint):
        self.logger.info(f'Start resume state dict from {checkpoint}')
        data = jt.load(checkpoint)
        model_state_dict = data['state_dict']
        self.model.load_state_dict(model_state_dict)

        # load train_loop info
        if self.train_loop is not None and 'loop' in data:
            self.train_loop.load_state_dict(data['loop'])

        # load optimizer info
        if self.optimizer is not None and 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])

        # load scheduler info
        if self.scheduler is not None and 'scheduler' in data:
            for scheduler in self.scheduler:
                scheduler_name = scheduler.__class__.__name__
                if scheduler_name in data['scheduler']:
                    scheduler.load_state_dict(
                        data['scheduler'][scheduler_name])
        self.logger.info('Finish resuming')

    def load(self, checkpoint):
        self.logger.info(f'Start load parameters from {checkpoint}')
        data = jt.load(checkpoint)
        model_state_dict = data['state_dict']
        self.model.load_state_dict(model_state_dict)
        self.logger.info('Finish loading checkpoint')

    @jt.single_process_scope()
    def save_checkpoint(self, filepath, end_epoch=True):
        data = dict()
        data['state_dict'] = self.model.state_dict()

        # train_loop info
        loop_state_dict = self.train_loop.state_dict()
        if end_epoch:
            loop_state_dict['epoch'] += 1
        data['loop'] = loop_state_dict

        # optimizer info
        data['optimizer'] = self.optimizer.state_dict()

        # scheduler info
        scheduler_state_dict = dict()
        for scheduler in self.scheduler:
            scheduler_name = scheduler.__class__.__name__
            scheduler_state_dict[scheduler_name] = scheduler.state_dict()
        data['scheduler'] = scheduler_state_dict
        jt.save(data, filepath)

    def setup_hooks(self, hooks):
        if hooks is None:
            hooks = []
        assert isinstance(hooks, list)

        default_hooks = [dict(type='LoggerHook'), dict(type='CheckpointHook')]
        hook_keys = [hook['type'] for hook in hooks]
        for hook in default_hooks:
            if hook['type'] not in hook_keys:
                hooks.append(hook)

        _hooks = []
        for hook in hooks:
            if not isinstance(hook, BaseHook):
                hook = HOOKS.build(hook)
            _hooks.append(hook)
        self._hooks = sorted(_hooks, key=lambda x: x.priority, reverse=True)

    def call_hook(self, func_name, **kwargs):
        for hook in self._hooks:
            getattr(hook, func_name)(self, **kwargs)
