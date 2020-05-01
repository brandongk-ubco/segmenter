from abc import abstractmethod, ABCMeta
import argparse
from tensorflow.keras import backend as K
import os


class BaseEvaluator(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute)
                and hasattr(subclass, 'evaluate_threshold')
                and callable(subclass.evaluate_threshold))

    def __init__(self, clazz: str, job_config, job_hash: str, datadir: str,
                 outdir: str, weight_finder, **kwargs):
        K.clear_session()
        self.job_config = job_config
        self.clazz = clazz
        self.job_hash = job_hash
        self.datadir = datadir
        self.outdir = outdir
        self.weight_finder = weight_finder(
            os.path.join(self.outdir, self.job_hash, self.clazz))
        self.resultdir = os.path.join(outdir, job_hash, clazz, "results")

    @abstractmethod
    def execute(self) -> None:
        raise NotImplementedError
