from abc import abstractmethod, ABCMeta
import argparse
from typing import Dict


class Task(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'arguments') and callable(subclass.arugments)
                and hasattr(subclass, 'arguments_to_cli')
                and callable(subclass.arguments_to_cli)
                and hasattr(subclass, 'execute')
                and callable(subclass.execute))

    @staticmethod
    @abstractmethod
    def arguments(arguments: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def arguments_to_cli(args: Dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, args) -> None:
        raise NotImplementedError
