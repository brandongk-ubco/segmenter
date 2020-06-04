from abc import abstractmethod, ABCMeta


class BaseCollector(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute))

    def __init__(self, data_dir: str, ground_truth_dir: str, job_config):
        self.data_dir = data_dir
        self.ground_truth_dir = ground_truth_dir
        self.job_config = job_config

    def execute(self) -> None:
        raise NotImplementedError