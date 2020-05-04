from enum import Enum


class FoldWeightFinders(Enum):
    best = "best"
    organized = "organized"
    latest = "latest"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    @staticmethod
    def get(finder):

        if finder == "best":
            from segmenter.models.BestFoldWeightFinder import BestFoldWeightFinder
            return BestFoldWeightFinder
        if finder == "organized":
            from segmenter.models.OrganizedFoldWeightFinder import OrganizedFoldWeightFinder
            return OrganizedFoldWeightFinder
        if finder == "latest":
            from segmenter.models.LatestFoldWeightFinder import LatestFoldWeightFinder
            return LatestFoldWeightFinder

        raise ValueError("Unknown fold weight finder {}".format(finder))
