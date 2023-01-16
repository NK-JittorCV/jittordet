from abc import ABCMeta, abstractmethod


class BaseEvaluator(metaclass=ABCMeta):

    @abstractmethod
    def evaluate(self, dataset, results):
        pass
