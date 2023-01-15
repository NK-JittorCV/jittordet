import copy
import inspect
import os
import os.path as osp
import time

import jittor as jt
import jittor.nn as nn
from jittor.dataset import Dataset
from jittor.optim import Optimizer

from .hooks import BaseHook
from .logger import get_logger
from .loops import BaseLoop
from .register import DATASETS, HOOKS, LOOPS, MODELS, OPTIMIZERS, SCHEDULERS

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
                 randomness=None,
                 resume_from=None,
                 load_from=None,
                 experiment_name=None,
                 cfg=None):
        # setup work dir
        self._work_dir = osp.abspath(work_dir)
        if not osp.exists(self._work_dir):
            os.makedirs(self._work_dir)

        # hold on original configs
        self.cfg = cfg
        if self.cfg is not None:
            self.cfg = copy.deepcopy(self.cfg)

        # set random seed and other randomness related setting
        self.setup_randomness(randomness)

        self._time_stamp = time.strftime('%Y%m%d_%H%M%S',
                                         time.localtime(time.item()))

        # setup experiment name
        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self._timestamp

        # setup hooks
        self.setup_hooks(hooks)

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
            randomness=cfg.get('randomness'),
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

    def init_model_weights(self, model):
        for module in model.children():
            if hasattr(module, 'init_weights'):
                module.init_weights()

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

            return DATASETS.build(dataset, drop_last=jt.in_mpi)
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

        return param_schedulers

    def build_loop(self, loop):
        if isinstance(loop, BaseLoop):
            return loop
        elif isinstance(loop, dict):
            return LOOPS.build(loop, runner=self)
        else:
            raise TypeError('loops should be a BaseLoop object or dict')

    def build_evaluator(self, evaluator):
        # if isinstance(evaluator, BaseEvaluator):
        #     return evaluator
        # elif isinstance(evaluator, dict):
        #     return EVALUATORS.build(evaluator, runner=self)
        # else:
        #     raise TypeError(
        #         'evaluator should be a BaseEvaluator object or dict')
        pass

    def train(self):
        log_dir = osp.join(self.work_dir, self.time_stamp + '_train')
        log_file = osp.join(log_dir, self.time_stamp + '.log')
        self.log_dir = log_dir
        self.logger = get_logger(name='jittordet', log_flie=log_file)

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
        self.init_model_weights()
        self.load_or_resume()

        # start run
        self.train_loop.run()

    def val(self):
        log_dir = osp.join(self.work_dir, self.time_stamp + '_val')
        log_file = osp.join(log_dir, self.time_stamp + '.log')
        self.log_dir = log_dir
        self.logger = get_logger(name='jittordet', log_flie=log_file)

        val_related = (self.val_dataset, self.val_loop, self.val_evaluator)
        assert all([item is not None for item in val_related]), \
            'In val phase, "val_dataset", "val_loop", and "val_evaluator" ' \
            'should be all None or not None.'
        self.val_dataset = self.build_dataset(self.val_dataset)
        self.val_loop = self.build_loop(self.val_loop)
        self.val_evaluator = self.build_evaluator(self.val_evaluator)

        # initialization
        self.load_or_resume()

        # start run
        self.val_loop.run()

    def test(self):
        log_dir = osp.join(self.work_dir, self.time_stamp + '_test')
        log_file = osp.join(log_dir, self.time_stamp + '.log')
        self.log_dir = log_dir
        self.logger = get_logger(name='jittordet', log_flie=log_file)

        test_related = (self.test_dataset, self.test_loop, self.test_evaluator)
        assert all([item is not None for item in test_related]), \
            'In test phase, "test_dataset", "test_loop", and ' \
            '"test_evaluator" should be all None or not None.'
        self.test_dataset = self.build_dataset(self.test_dataset)
        self.test_loop = self.build_loop(self.test_loop)
        self.test_evaluator = self.build_evaluator(self.test_evaluator)

        # initialization
        self.load_or_resume()

        # start run
        self.val_loop.run()

    def setup_randomness(self, cfg):
        pass

    def resume(self):
        pass

    def load(self):
        pass

    def save(self, checkpoint_path):
        pass

    def setup_hooks(self, hooks):
        assert isinstance(hooks, list)
        default_hooks = [dict(type='LoggerHook'), dict(type='CheckPointHook')]
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
