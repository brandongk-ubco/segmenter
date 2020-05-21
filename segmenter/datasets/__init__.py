from enum import Enum


class Datasets(Enum):
    severstal = "severstal"
    kits = "kits19"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return sorted([e.value for e in cls])

    @staticmethod
    def get(dataset):
        if dataset == "severstal":
            from segmenter.datasets.severstal.SeverstalDataset import SeverstalDataset
            return SeverstalDataset
        if dataset == "kits19":
            from segmenter.datasets.kits19.KitsDataset import KitsDataset
            return KitsDataset

        raise ValueError("Unknown evaluator {}".format(dataset))
