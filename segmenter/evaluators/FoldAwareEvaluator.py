from abc import abstractmethod, ABCMeta
from segmenter.evaluators.BaseEvaluator import BaseEvaluator
from tensorflow.keras import backend as K
import os
from segmenter.data import augmented_generator
from segmenter.augmentations import predict_augments
from segmenter.optimizers import get_optimizer
from segmenter.loss import get_loss
from segmenter.models.full_model import model_for_fold, model_for_boost_fold
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import itertools


class FoldAwareEvaluator(BaseEvaluator, metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return BaseEvaluator.__subclasshook__(subclass) and hasattr(
            subclass, 'evaluate_fold') and callable(subclass.evaluate_fold)

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
        self.classes = self.job_config["CLASSES"]
        self.folds = kwargs.get("folds")

    def execute(self) -> None:
        for fold_name in self.folds:
            result_dir = os.path.join(self.resultdir, fold_name)
            os.makedirs(result_dir, exist_ok=True)
            inputs = Input(shape=(None, None, 1))
            if "b" in fold_name:
                model = model_for_boost_fold(fold_name, self.job_config,
                                             self.weight_finder)
                model = model(inputs)
                model = Model(inputs, model)
            else:
                model = model_for_fold(fold_name, self.job_config,
                                       self.weight_finder, "sigmoid", inputs)
            model.summary()
            self.evaluate_fold(fold_name, model, result_dir)

    @abstractmethod
    def evaluate_fold(self, fold_name, fold_model, result_dir) -> None:
        raise NotImplementedError
