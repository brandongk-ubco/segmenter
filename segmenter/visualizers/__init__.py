from enum import Enum


class Visualizers(Enum):
    auc = "auc"
    predict = "predict"
    activation = "activation"
    report = "report"
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

        if visualizer == "arc":
            from segmenter.visualizers.AUCVisualizer import AUCVisualizer
            return AUCVisualizer
        if visualizer == "predict":
            from segmenter.visualizers.PredictionVisualizer import PredictionVisualizer
            return PredictionVisualizer
        if visualizer == "activation":
            from segmenter.visualizers.ActivationVisualizer import ActivationVisualizer
            return ActivationVisualizer
        if visualizer == "report":
            from segmenter.visualizers.ReportVisualizer import ReportVisualizer
            return ReportVisualizer
        if visualizer == "metric":
            from segmenter.visualizers.MetricVisualizer import MetricVisualizer
            return MetricVisualizer
        if visualizer == "boost":
            from segmenter.visualizers.BoostVisualizer import BoostVisualizer
            return BoostVisualizer

        raise ValueError("Unknown visualizer {}".format(visualizer))
