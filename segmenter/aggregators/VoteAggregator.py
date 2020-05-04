from segmenter.layers import Vote
from segmenter.aggregators.Aggregator import Aggregator
import numpy as np


class VoteAggregator(Aggregator):
    def name(self):
        return "vote"

    def display_name(self):
        return "Vote"

    def thresholds(self):
        num_folds = self.job_config["FOLDS"]
        return [0] if num_folds == 0 else np.linspace(
            0., (num_folds - 1) / num_folds, num=num_folds)

    def layer(self):
        return Vote

    def fold_activation(self):
        return "sigmoid"

    def final_activation(self):
        return "linear"
