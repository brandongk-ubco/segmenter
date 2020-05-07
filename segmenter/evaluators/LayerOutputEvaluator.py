from tensorflow.keras import backend as K
from segmenter.evaluators.FoldAwareEvaluator import FoldAwareEvaluator
from tensorflow.keras.models import Model
from segmenter.loss import get_loss
from segmenter.data import augmented_generator
from segmenter.augmentations import predict_augments
from segmenter.aggregators import Aggregator
from segmenter.optimizers import get_optimizer
import numpy as np
import os


class LayerOutputEvaluator(FoldAwareEvaluator):

    layer_types = ["activation", "convolutional", "normalization"]
    bins = np.linspace(-10, 10, num=2001)

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

    def evaluate_fold(self, fold_name, fold_model, result_dir):

        fold_results_file = os.path.join(result_dir, "layer_outputs")
        if os.path.exists("{}.npz".format(fold_results_file)):
            print("Fold {} already processed".format(fold_name))
            return

        fold_results = dict([(l,
                              np.zeros(np.histogram([],
                                                    bins=self.bins)[0].shape,
                                       dtype="uint64"))
                             for l in self.layer_types])

        for layer_type in self.layer_types:
            all_outputs = LayerOutputEvaluator.get_outputs(fold_model,
                                                           matcher=layer_type)
            num_layers = len(all_outputs[0][1])
            for m, outputs in all_outputs:
                for batch, (images, _masks) in enumerate(self.dataset):
                    print("{} {} layers for {} - image {}/{}".format(
                        num_layers, layer_type, m.name, batch + 1,
                        self.num_images))
                    for o in Model(m.input, outputs).predict_on_batch(images):
                        fold_results[layer_type] += np.histogram(
                            o, bins=self.bins)[0].astype("uint64")

        np.savez_compressed(fold_results_file, **fold_results)
