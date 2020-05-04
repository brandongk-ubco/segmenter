import numpy as np
from segmenter.aggregators.Aggregator import Aggregator
from segmenter.layers import NoisyOr, Vote
from tensorflow.keras.layers import Average


class NoisyOrAggregator(Aggregator):
    def name(self):
        return "noisy_or"

    def thresholds(self):
        return np.append(np.linspace(0.5, 0.75, num=6),
                         np.linspace(0.8, 0.99, num=20))

    def layer(self):
        return NoisyOr

    def fold_activation(self):
        return "sigmoid"

    def final_activation(self):
        return "linear"


class VoteAggregator(Aggregator):
    def __init__(self, num_folds):
        self.num_folds = num_folds

    def name(self):
        return "vote"

    def thresholds(self):
        return [0] if self.num_folds == 0 else np.linspace(
            0., (self.num_folds - 1) / self.num_folds, num=self.num_folds)

    def layer(self):
        return Vote

    def fold_activation(self):
        return "sigmoid"

    def final_activation(self):
        return "linear"


class AverageAggregator(Aggregator):
    def name(self):
        return "average"

    def thresholds(self):
        return np.linspace(0., 0.95, num=20)

    def layer(self):
        return Average

    def fold_activation(self):
        return "linear"

    def final_activation(self):
        return "sigmoid"


class DummyAggregator(Aggregator):
    def name(self):
        return "dummy"

    def thresholds(self):
        return np.linspace(0., 0.95, num=20)

    def layer(self):
        return Activation("linear")

    def fold_activation(self):
        return "linear"

    def final_activation(self):
        return "sigmoid"


def get_aggregators(job_config, aggregators=None):
    if job_config["FOLDS"] > 0:
        return [
            VoteAggregator(job_config["FOLDS"]),
            NoisyOrAggregator(),
            AverageAggregator()
        ]
    else:
        return [DummyAggregator()]
