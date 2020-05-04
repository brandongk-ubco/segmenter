from abc import abstractmethod, ABCMeta
import argparse


class BaseVisualizer(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute))

    def __init__(self, data_dir: str, job_config):
        self.data_dir = data_dir
        self.job_config = job_config

    def execute(self) -> None:
        raise NotImplementedError