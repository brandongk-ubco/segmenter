from tensorflow.keras import backend as K
from segmenter.evaluators.FoldAwareEvaluator import FoldAwareEvaluator
from tensorflow.keras.models import Model
from segmenter.models import full_model
from segmenter.loss import get_loss
from segmenter.data import augmented_generator
from segmenter.augmentations import predict_augments
from segmenter.aggregators import Aggregator
from segmenter.optimizers import get_optimizer
from segmenter.models import full_model
import numpy as np
import os


class ActivationEvaluator(FoldAwareEvaluator):

    layer_types = ["activation", "convolutional", "normalization"]
    bins = np.linspace(-10, 10, num=2001)

    def __init__(self, *args, **kwargs):
        super(ActivationEvaluator, self).__init__(*args, **kwargs)
        self.results = dict([(l,
                              np.zeros(np.histogram([],
                                                    bins=self.bins)[0].shape,
                                       dtype="uint64"))
                             for l in self.layer_types])

    @staticmethod
    def get_outputs(model, matcher):
        outputs = []
        for m in [
                layer for layer in model.layers if isinstance(layer, (Model))
        ]:
            outputs.append((m, [
                l.output for l in m.layers if matcher in str(type(l)).lower()
            ]))
        return outputs

    def evaluate_fold(self, fold_model):
        for batch, (images, _masks) in enumerate(self.dataset):
            print("{} ({}/{})".format(fold_model.name, batch, self.num_images))
            fold_model.predict_on_batch(images).numpy()

            for layer_type in self.layer_types:
                all_outputs = ActivationEvaluator.get_outputs(
                    fold_model, matcher=layer_type)
                for i, (m, outputs) in enumerate(all_outputs):
                    for batch, (images, _masks) in enumerate(self.dataset):
                        print("{} model {} {}/{} image {}/{}".format(
                            layer_type, m.name, i + 1, len(all_outputs),
                            batch + 1, self.num_images))
                        for o in Model(m.input,
                                       outputs).predict_on_batch(images):
                            self.results[layer_type] += np.histogram(
                                o, bins=self.bins)[0].astype("uint64")

    def after_execution(self):
        np.savez_compressed(os.path.join(self.resultdir, "activation"),
                            **self.results)
