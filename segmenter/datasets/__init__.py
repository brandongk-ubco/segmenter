from segmenter.datasets.severstal.SeverstalDataset import SeverstalDataset
from segmenter.datasets.kits.KitsDataset import KitsDataset
from enum import Enum


class Datasets(Enum):
    severstal = SeverstalDataset
    kits = KitsDataset

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Datasets[s]
        except KeyError:
            return s
