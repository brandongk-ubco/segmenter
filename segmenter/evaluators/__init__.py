from enum import Enum


class Evaluators(Enum):
    metric = "metric"
    predict = "predict"
    layer_output = "layer-output"
    variance = "variance"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    @staticmethod
    def get(evaluator):

        if evaluator == "metric":
            from segmenter.evaluators.MetricEvaluator import MetricEvaluator
            return MetricEvaluator
        if evaluator == "predict":
            from segmenter.evaluators.PredictEvaluator import PredictEvaluator
            return PredictEvaluator
        if evaluator == "layer-output":
            from segmenter.evaluators.LayerOutputEvaluator import LayerOutputEvaluator
            return LayerOutputEvaluator
        if evaluator == "variance":
            from segmenter.evaluators.VarianceEvaluator import VarianceEvaluator
            return VarianceEvaluator

        raise ValueError("Unknown evaluator {}".format(evaluator))
