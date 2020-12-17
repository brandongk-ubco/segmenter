from segmenter.aggregators.Aggregator import Aggregator
import numpy as np


class DummyAggregator(Aggregator):
    def name(self):
        return "dummy"

    def thresholds(self):
        # return np.linspace(0., 0.9, num=10)
        return [0.5]

    def display_name(self):
        return "Dummy"

    def layer(self):
        raise NotImplementedError

    def fold_activation(self):
        return "linear"

    def final_activation(self):
        return "sigmoid"
