import numpy as np
from segmenter.aggregators.Aggregator import Aggregator
from segmenter.layers import NoisyOr


class NoisyOrAggregator(Aggregator):
    @staticmethod
    def name():
        return "noisy_or"

    @staticmethod
    def thresholds():
        return np.append(np.linspace(0.5, 0.75, num=6),
                         np.linspace(0.8, 0.99, num=20))

    @staticmethod
    def layer():
        return NoisyOr

    @staticmethod
    def fold_activation():
        return "sigmoid"

    @staticmethod
    def final_activation():
        return "linear"


class VoteAggregator(Aggregator):
    pass


class AverageAggregator(Aggregator):
    pass


def get_aggregators(job_config, aggregators=None):
    return [NoisyOrAggregator]
    # num_folds = job_config.get("FOLDS", 0)
    # ag = [(
    #     "vote",
    #     Vote,
    #     "sigmoid",
    #     "linear",
    # ),
    #       ("leaky_noisy_or", LeakyNoisyOr, "sigmoid", "linear",
    #        np.append(np.linspace(0.5, 0.75, num=6),
    #                  np.linspace(0.8, 0.99, num=20))),
    #       ("average", Average, "linear", "sigmoid",
    #        np.linspace(0., 0.95, num=20))]
    # if aggregators is not None:
    #     ag = list(filter(lambda x: x[0] in aggregators, ag))
    # return ag