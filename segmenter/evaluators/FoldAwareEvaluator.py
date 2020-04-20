from abc import abstractmethod, ABCMeta
from segmenter.evaluators.BaseEvaluator import BaseEvaluator
from tensorflow.keras import backend as K
import os
from segmenter.aggregators import Aggregator
from segmenter.data import augmented_generator
from segmenter.augmentations import predict_augments
from segmenter.optimizers import get_optimizer
from segmenter.loss import get_loss
from segmenter.models import model_folds
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class FoldAwareEvaluator(BaseEvaluator, metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return BaseEvaluator.__subclasshook__(subclass) and hasattr(
            subclass,
            'evaluate_fold') and callable(subclass.evaluate_fold) and hasattr(
                subclass, 'after_execution') and callable(
                    subclass.after_execution)

    def __init__(self, *args, **kwargs):
        super(FoldAwareEvaluator, self).__init__(*args, **kwargs)
        K.clear_session()

        self.generator, self.dataset, self.num_images = augmented_generator(
            self.clazz, None, predict_augments, self.job_config, "evaluate",
            self.datadir)
        self.dataset = self.dataset.batch(1)

        self.loss = get_loss(self.job_config["LOSS"])
        self.inputs = Input(shape=(None, None, 1))
        self.optimizer = get_optimizer(self.job_config["OPTIMIZER"])
        self.models = model_folds(self.inputs, self.clazz, self.outdir,
                                  self.job_config, self.job_hash, "linear")

    def execute(self) -> None:
        for model in self.models:
            model.summary()
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=[])
            self.evaluate_fold(model)

    @abstractmethod
    def evaluate_fold(self, fold_model) -> None:
        raise NotImplementedError

    @abstractmethod
    def after_execution(self) -> None:
        pass