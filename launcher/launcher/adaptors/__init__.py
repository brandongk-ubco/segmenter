from .Shell import ShellAdaptor
from .ComputeCanada import ComputeCanadaAdaptor
from enum import Enum


class Adaptors(Enum):
    shell = ShellAdaptor
    computecanada = ComputeCanadaAdaptor

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def argparse(s):
        try:
            return Adaptors[s]
        except KeyError:
            return s
