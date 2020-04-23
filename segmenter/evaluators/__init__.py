from enum import Enum


class Evaluators(Enum):
    metric = "metric"
    predict = "predict"
    activations = "activation"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return [e.value for e in cls]

    @staticmethod
    def get(evaluator):

        if evaluator == "metric":
            from segmenter.evaluators.MetricEvaluator import MetricEvaluator
            return MetricEvaluator
        if evaluator == "predict":
            from segmenter.evaluators.PredictEvaluator import PredictEvaluator
            return PredictEvaluator
        if evaluator == "activation":
            from segmenter.evaluators.ActivationEvaluator import ActivationEvaluator
            return ActivationEvaluator

        raise ValueError("Unknown evaluator {}".format(evaluator))
