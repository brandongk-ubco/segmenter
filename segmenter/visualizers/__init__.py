from enum import Enum
from segmenter.visualizers.VarianceVisualizer import VarianceVisualizer


class Visualizers(Enum):
    auc = "auc"
    combined_auc = "combined-auc"
    predict = "predict"
    layer_output = "layer-output"
    combined_layer_output = "combined-layer-output"
    instance_metrics = "instance-metrics"
    metric = "metric"
    combined_f1 = "combined-f1"
    boost = "boost"
    best_threshold = "best-threshold"
    confusion = "confusion"
    seach_depth_width = "search-depth-width"
    search_parallel_coordinates = "search-parallel-coordinates"
    search_activation = "search-activation"
    variance = "variance"
    best_pair = "best-pair"

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
        if visualizer == "combined-auc":
            from segmenter.visualizers.CombinedAUCVisualizer import CombinedAUCVisualizer
            return CombinedAUCVisualizer
        if visualizer == "predict":
            from segmenter.visualizers.PredictionVisualizer import PredictionVisualizer
            return PredictionVisualizer
        if visualizer == "layer-output":
            from segmenter.visualizers.LayerOutputVisualizer import LayerOutputVisualizer
            return LayerOutputVisualizer
        if visualizer == "combined-layer-output":
            from segmenter.visualizers.CombinedLayerOutputVisualizer import CombinedLayerOutputVisualizer
            return CombinedLayerOutputVisualizer
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
        if visualizer == "combined-f1":
            from segmenter.visualizers.CombinedF1Visualizer import CombinedF1Visualizer
            return CombinedF1Visualizer
        if visualizer == "search-depth-width":
            from segmenter.visualizers.SearchDepthWidthVisualizer import SearchDepthWidthVisualizer
            return SearchDepthWidthVisualizer
        if visualizer == "search-parallel-coordinates":
            from segmenter.visualizers.SearchParallelCoordinatesVisualizer import SearchParallelCoordinatesVisualizer
            return SearchParallelCoordinatesVisualizer
        if visualizer == "search-activation":
            from segmenter.visualizers.SearchActivationVisualizer import SearchActivationVisualizer
            return SearchActivationVisualizer
        if visualizer == "variance":
            from segmenter.visualizers.VarianceVisualizer import VarianceVisualizer
            return VarianceVisualizer
        if visualizer == "best-pair":
            from segmenter.visualizers.BestPairVisualizer import BestPairVisualizer
            return BestPairVisualizer

        raise ValueError("Unknown visualizer {}".format(visualizer))
