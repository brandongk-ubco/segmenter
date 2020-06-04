from abc import abstractmethod, ABCMeta
import argparse


class BaseVisualizer(metaclass=ABCMeta):

    label = ""

    label_map = {
        "8ba8ae882c396d08aaa3332167cd7aeb": "Weak Learners",
        "c4fc907d76c4adc975169e34d32b95df": "Boosted Learners",
        "c041f9d41605254f805232999f143ab0": "Baseline",
        "f9e2afd0b43fb3f43ab7dc2a95bd6368": "Weak Learners",
        "54e053312ee47fe203400e055fa93be8": "Boosted Learners",
        "89489ee3b0c0504c424df68d1672f0cf": "Baseline",
        "4e4c5e07f5728b9b5d7ec365b8e211c8": "Strong Learners",
        "89030ed7105f2d5c95654f5429366a55": "Strong Learners",
    }

    job_combined_visualizer = False
    dataset_combined_visualizer = False
    full_combined_visualizer = False

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute))

    def __init__(self, data_dir: str, job_config, job_hash: str):
        self.data_dir = data_dir
        self.job_config = job_config
        self.job_hash = job_hash
        if self.job_hash is not None:
            self.label = self.label_map[self.job_hash]

    def execute(self) -> None:
        raise NotImplementedError