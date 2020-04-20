from tensorflow.keras import backend as K
from segmenter.augmentations import predict_augments
from segmenter.callbacks import get_evaluation_callbacks
import json
import os
from segmenter.evaluators.ThresholdAwareEvaluator import ThresholdAwareEvaluator


class MetricEvaluator(ThresholdAwareEvaluator):
    def evaluate_threshold(self, model, threshold, outdir):
        results = model.evaluate(x=self.dataset,
                                 callbacks=get_evaluation_callbacks(),
                                 verbose=1,
                                 steps=self.num_images)

        with open(os.path.join(outdir, "results.json"), "w") as results_json:
            results_dict = dict(
                zip(['loss'] + list(self.metrics.keys()),
                    [float(r) for r in results]))
            json.dump(results_dict, results_json)
