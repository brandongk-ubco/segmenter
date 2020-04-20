from segmenter.visualizers.AUCVisualizer import AUCVisualizer
from segmenter.visualizers.PredictionVisualizer import PredictionVisualizer
from segmenter.visualizers.ActivationVisualizer import ActivationVisualizer
from enum import Enum


class Visualizers(Enum):
    auc = AUCVisualizer
    predict = PredictionVisualizer
    activations = ActivationVisualizer

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Visualizers[s]
        except KeyError:
            return s
