from abc import abstractmethod, ABCMeta
import argparse


class Task(metaclass=ABCMeta):

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
    def arguments(arguments: argparse._SubParsersAction) -> None:
        pass

    @abstractmethod
    def execute(self, args) -> None:
        pass
