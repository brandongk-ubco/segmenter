from abc import abstractmethod, ABCMeta
import argparse
from typing import Dict, Any
from launcher import Task


class Adaptor(metaclass=ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'arguments') and
            callable(subclass.arugments) and
            hasattr(subclass, 'execute') and
            callable(subclass.execute)
        )

    @staticmethod
    @abstractmethod
    def arguments(arguments: argparse.ArgumentParser) -> None:
        pass

    @staticmethod
    @abstractmethod
    def execute(task: Task, args: Dict) -> None:
        pass
