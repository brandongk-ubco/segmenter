from enum import Enum


class Visualizers(Enum):
    auc = "auc"
    predict = "predict"
    activation = "activation"
    instance_metrics = "instance-metrics"
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

        if visualizer == "auc":
            from segmenter.visualizers.AUCVisualizer import AUCVisualizer
            return AUCVisualizer
        if visualizer == "predict":
            from segmenter.visualizers.PredictionVisualizer import PredictionVisualizer
            return PredictionVisualizer
        if visualizer == "activation":
            from segmenter.visualizers.ActivationVisualizer import ActivationVisualizer
            return ActivationVisualizer
        if visualizer == "instance-metrics":
            from segmenter.visualizers.InstanceMetricsVisualizer import InstanceMetricsVisualizer
            return InstanceMetricsVisualizer
        if visualizer == "metric":
            from segmenter.visualizers.MetricVisualizer import MetricVisualizer
            return MetricVisualizer
        if visualizer == "boost":
            from segmenter.visualizers.BoostVisualizer import BoostVisualizer
            return BoostVisualizer

        raise ValueError("Unknown visualizer {}".format(visualizer))
