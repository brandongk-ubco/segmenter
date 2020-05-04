from enum import Enum


class Collectors(Enum):
    predict = "predict"
    metric = "metric"
    boost = "boost"
    wight = "weight"
    instance_metrics = "instance-metrics"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    @staticmethod
    def get(collector):

        if collector == "predict":
            from segmenter.collectors.PredictionCollector import PredictionCollector
            return PredictionCollector
        if collector == "metric":
            from segmenter.collectors.MetricCollector import MetricCollector
            return MetricCollector
        if collector == "boost":
            from segmenter.collectors.BoostCollector import BoostCollector
            return BoostCollector
        if collector == "weight":
            from segmenter.collectors.WeightCollector import WeightCollector
            return WeightCollector
        if collector == "instance-metrics":
            from segmenter.collectors.InstanceMetricsCollector import InstanceMetricsCollector
            return InstanceMetricsCollector

        raise ValueError("Unknown collector {}".format(collector))
