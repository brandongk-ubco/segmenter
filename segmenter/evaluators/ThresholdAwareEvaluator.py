from abc import abstractmethod, ABCMeta
from segmenter.evaluators.BaseEvaluator import BaseEvaluator
from tensorflow.keras import backend as K
import os
from segmenter.aggregators import Aggregator
from segmenter.data import augmented_generator
from segmenter.augmentations import predict_augments
from segmenter.loss import get_loss
from segmenter.models.full_model import full_model, model_for_fold
from segmenter.optimizers import get_optimizer
from segmenter.metrics import get_metrics
from segmenter.aggregators import Aggregators
import numpy as np


class ThresholdAwareEvaluator(BaseEvaluator, metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return BaseEvaluator.__subclasshook__(subclass) and hasattr(
            subclass, 'evaluate_threshold') and callable(
                subclass.evaluate_threshold)

    def __init__(self, *args, **kwargs):
        super(ThresholdAwareEvaluator, self).__init__(*args, **kwargs)
        K.clear_session()

        self.generator, self.dataset, self.num_images = augmented_generator(
            self.clazz, None, predict_augments, self.job_config, "evaluate",
            self.datadir)
        self.dataset = self.dataset.batch(1)

        self.loss = get_loss(self.job_config["LOSS"])

        self.optimizer = get_optimizer(self.job_config["OPTIMIZER"])

    def execute(self) -> None:
        for aggregator_name in Aggregators.choices():
            K.clear_session()
            aggregator = Aggregators.get(aggregator_name)(self.job_config)
            model = full_model(self.job_config,
                               self.weight_finder,
                               aggregator=aggregator)
            for threshold in aggregator.thresholds():
                threshold_str = "{:1.2f}".format(threshold)
                print("Aggregator {} and Threshold: {}".format(
                    aggregator.display_name(), threshold_str))
                threshold_dir = os.path.join(self.resultdir, aggregator.name(),
                                             threshold_str)
                os.makedirs(threshold_dir, exist_ok=True)
                self.metrics = get_metrics(threshold, self.job_config["LOSS"])
                model.compile(optimizer=self.optimizer,
                              loss=self.loss,
                              metrics=list(self.metrics.values()))
                self.evaluate_threshold(model, threshold, threshold_dir)

    @abstractmethod
    def evaluate_threshold(self, model, threshold, threshold_dir) -> None:
        raise NotImplementedError
