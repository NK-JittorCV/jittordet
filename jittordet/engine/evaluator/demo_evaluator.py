from ..register import EVALUATORS
from .base_evaluator import BaseEvaluator


@EVALUATORS.register_module()
class DemoEvaluator(BaseEvaluator):

    def evaluate(self, datasets, results):
        metrics = dict(mAP=0.9)
        return metrics
