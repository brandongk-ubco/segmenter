from abc import abstractmethod, ABCMeta
from typing import List


class FoldWeightFinder(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'get') and callable(subclass.get))

    @abstractmethod
    def get(self, fold_name) -> str:
        raise NotImplementedError

    @abstractmethod
    def keys(self) -> List[str]:
        raise NotImplementedError
