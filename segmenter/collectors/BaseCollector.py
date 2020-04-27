from abc import abstractmethod, ABCMeta


class BaseCollector(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute))

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def execute(self) -> None:
        raise NotImplementedError