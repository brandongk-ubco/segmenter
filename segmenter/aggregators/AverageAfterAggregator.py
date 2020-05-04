from segmenter.aggregators.Aggregator import Aggregator
import numpy as np
from tensorflow.keras.layers import Average


class AverageAfterAggregator(Aggregator):
    def name(self):
        return "average_after"

    def thresholds(self):
        return np.append(
            np.append(np.linspace(0., 0.35, num=8),
                      np.linspace(0.36, 0.45, num=10)),
            np.linspace(0.5, 0.95, num=10))

    def display_name(self):
        return "Average After Sigmoid"

    def layer(self):
        return Average

    def fold_activation(self):
        return "sigmoid"

    def final_activation(self):
        return "linear"
