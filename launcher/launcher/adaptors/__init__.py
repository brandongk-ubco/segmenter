from launcher.adaptors.ShellAdaptor import ShellAdaptor
from launcher.adaptors.DockerAdaptor import DockerAdaptor
from launcher.adaptors.SingularityAdaptor import SingularityAdaptor
from enum import Enum


class Adaptors(Enum):
    shell = ShellAdaptor
    docker = DockerAdaptor
    singularity = SingularityAdaptor

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
