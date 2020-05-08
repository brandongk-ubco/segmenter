from abc import abstractmethod, ABCMeta
import argparse


class BaseVisualizer(metaclass=ABCMeta):

    label = ""

    label_map = {
        "8ba8ae882c396d08aaa3332167cd7aeb": "Weak Learner",
        "c4fc907d76c4adc975169e34d32b95df": "Boosted Learners",
        "c041f9d41605254f805232999f143ab0": "Baseline"
    }

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute))

    def __init__(self, data_dir: str, job_config):
        self.data_dir = data_dir
        self.job_config = job_config
        self.job_hash = data_dir.split("/")[-3]
        self.label = self.label_map[self.job_hash]

    def execute(self) -> None:
        raise NotImplementedError