from segmenter.evaluators.MetricEvaluator import MetricEvaluator
from segmenter.evaluators.PredictEvaluator import PredictEvaluator
from segmenter.evaluators.ActivationEvaluator import ActivationEvaluator
from enum import Enum


class Evaluators(Enum):
    metric = MetricEvaluator
    predict = PredictEvaluator
    activations = ActivationEvaluator

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Evaluators[s]
        except KeyError:
            return s
