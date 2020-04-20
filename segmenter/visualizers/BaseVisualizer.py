from abc import abstractmethod, ABCMeta
import argparse
from typing import Dict, Any
from tensorflow.keras import backend as K
from segmenter.data import augmented_generator
from segmenter.models import full_model
from segmenter.optimizers import get_optimizer
from segmenter.aggregators import Aggregator
from segmenter.loss import get_loss
from segmenter.augmentations import predict_augments
from segmenter.metrics import get_metrics
import os


class BaseVisualizer(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute))

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def execute(self) -> None:
        raise NotImplementedError