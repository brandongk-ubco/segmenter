from abc import abstractmethod, ABCMeta
import numpy as np
from tensorflow.python.keras.engine.base_layer import Layer


class Aggregator:
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'name') and callable(subclass.name)
                and hasattr(subclass, 'thresholds')
                and callable(subclass.thresholds)
                and hasattr(subclass, 'layer') and callable(subclass.layer)
                and hasattr(subclass, 'fold_activation')
                and callable(subclass.fold_activation)
                and hasattr(subclass, 'final_activation')
                and callable(subclass.final_activation)
                and hasattr(subclass, 'display_name')
                and callable(subclass.display_name))

    def __init__(self, job_config):
        self.job_config = job_config

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def thresholds(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def display_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def layer(self) -> Layer:
        raise NotImplementedError

    @abstractmethod
    def fold_activation(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def final_activation(self) -> str:
        raise NotImplementedError
