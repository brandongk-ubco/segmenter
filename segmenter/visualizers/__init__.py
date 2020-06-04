from enum import Enum


class Visualizers(Enum):
    auc = "auc"
    predict = "predict"
    layer_output = "layer-output"
    instance_metrics = "instance-metrics"
    metric = "metric"
    boost = "boost"
    best_threshold = "best-threshold"
    confusion = "confusion"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    @staticmethod
    def get(visualizer):

        if visualizer == "auc":
            from segmenter.visualizers.AUCVisualizer import AUCVisualizer
            return AUCVisualizer
        if visualizer == "predict":
            from segmenter.visualizers.PredictionVisualizer import PredictionVisualizer
            return PredictionVisualizer
        if visualizer == "layer-output":
            from segmenter.visualizers.LayerOutputVisualizer import LayerOutputVisualizer
            return LayerOutputVisualizer
        if visualizer == "instance-metrics":
            from segmenter.visualizers.InstanceMetricsVisualizer import InstanceMetricsVisualizer
            return InstanceMetricsVisualizer
        if visualizer == "metric":
            from segmenter.visualizers.MetricVisualizer import MetricVisualizer
            return MetricVisualizer
        if visualizer == "boost":
            from segmenter.visualizers.BoostVisualizer import BoostVisualizer
            return BoostVisualizer
        if visualizer == "best-threshold":
            from segmenter.visualizers.BestThresholdVisualizer import BestThresholdVisualizer
            return BestThresholdVisualizer
        if visualizer == "confusion":
            from segmenter.visualizers.ConfusionVisualizer import ConfusionVisualizer
            return ConfusionVisualizer

        raise ValueError("Unknown visualizer {}".format(visualizer))
