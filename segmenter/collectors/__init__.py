from enum import Enum


class Collectors(Enum):
    predict = "predict"
    metric = "metric"
    boost = "boost"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return [e.value for e in cls]

    @staticmethod
    def get(visualizer):

        if visualizer == "predict":
            from segmenter.collectors.PredictionCollector import PredictionCollector
            return PredictionCollector
        if visualizer == "metric":
            from segmenter.collectors.MetricCollector import MetricCollector
            return MetricCollector
        if visualizer == "boost":
            from segmenter.collectors.BoostCollector import BoostCollector
            return BoostCollector

        raise ValueError("Unknown collector {}".format(visualizer))
