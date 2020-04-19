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
                and callable(subclass.final_activation))

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def thresholds() -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def layer() -> Layer:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def fold_activation() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def final_activation() -> str:
        raise NotImplementedError
