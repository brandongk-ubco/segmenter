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


class BaseEvaluator(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'execute') and callable(subclass.execute))

    def __init__(self, clazz: str, job_config, job_hash: str, datadir: str,
                 outdir: str, aggregator: Aggregator):
        K.clear_session()
        self.job_config = job_config

        self.aggregator = aggregator

        self.generator, self.dataset, self.num_images = augmented_generator(
            clazz, None, predict_augments, job_config, "evaluate", datadir)
        self.dataset = self.dataset.batch(1)

        self.resultdir = os.path.join(outdir, job_hash, clazz, "results",
                                      aggregator.name())
        self.loss = get_loss(job_config["LOSS"])

        self.optimizer = get_optimizer(self.job_config["OPTIMIZER"])

        self.model = full_model(clazz,
                                outdir,
                                job_config,
                                job_hash,
                                aggregator=aggregator.layer(),
                                fold_activation=aggregator.fold_activation(),
                                final_activation=aggregator.final_activation())

    def execute(self) -> None:
        for threshold in self.aggregator.thresholds():
            threshold_str = "{:1.2f}".format(threshold)
            print("Aggregator {} and Threshold: {}".format(
                self.aggregator.name(), threshold_str))
            threshold_dir = os.path.join(self.resultdir, threshold_str)
            os.makedirs(threshold_dir, exist_ok=True)
            self.metrics = get_metrics(threshold, self.job_config["LOSS"])
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=list(self.metrics.values()))
            self.evaluate_threshold(threshold, threshold_dir)

    @abstractmethod
    def evaluate_threshold(self, threshold, threshold_dir) -> None:
        raise NotImplementedError