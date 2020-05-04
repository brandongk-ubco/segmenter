from segmenter.aggregators.Aggregator import Aggregator
import numpy as np
from segmenter.layers import NoisyOr


class NoisyOrAggregator(Aggregator):
    def name(self):
        return "noisy_or"

    def thresholds(self):
        return np.append(np.linspace(0., 0.9, num=19),
                         np.linspace(0.91, 0.99, num=9))

    def display_name(self):
        return "Noisy Or"

    def layer(self):
        return NoisyOr

    def fold_activation(self):
        return "sigmoid"

    def final_activation(self):
        return "linear"
