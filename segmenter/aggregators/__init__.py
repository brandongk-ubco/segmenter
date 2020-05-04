import numpy as np
from segmenter.aggregators.Aggregator import Aggregator
from enum import Enum


class Aggregators(Enum):
    average_after = "average-after"
    average_before = "average-before"
    noisy_or = "noisy-or"
    vote = "vote"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    @staticmethod
    def get(aggregator):

        if aggregator == "noisy-or":
            from segmenter.aggregators.NoisyOrAggregator import NoisyOrAggregator
            return NoisyOrAggregator
        if aggregator == "average-before":
            from segmenter.aggregators.AverageBeforeAggregator import AverageBeforeAggregator
            return AverageBeforeAggregator
        if aggregator == "average-after":
            from segmenter.aggregators.AverageAfterAggregator import AverageAfterAggregator
            return AverageAfterAggregator
        if aggregator == "vote":
            from segmenter.aggregators.VoteAggregator import VoteAggregator
            return VoteAggregator

        raise ValueError("Unknown aggregator {}".format(aggregator))
