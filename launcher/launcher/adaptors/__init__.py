from enum import Enum


class Adaptors(Enum):
    shell = "shell"
    docker = "docker"
    singularity = "singularity"

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @classmethod
    def choices(cls):
        return [e.value for e in cls]

    @staticmethod
    def get(adaptor):
        if adaptor == "shell":
            from launcher.adaptors.ShellAdaptor import ShellAdaptor
            return ShellAdaptor
        if adaptor == "docker":
            from launcher.adaptors.DockerAdaptor import DockerAdaptor
            return DockerAdaptor
        if adaptor == "singularity":
            from launcher.adaptors.SingularityAdaptor import SingularityAdaptor
            return SingularityAdaptor

        raise ValueError("Unknown adaptor {}".format(adaptor))
