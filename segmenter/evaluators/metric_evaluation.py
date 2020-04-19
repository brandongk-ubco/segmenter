from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from segmenter.config import get_config
from segmenter.metrics import get_metrics
from segmenter.models import full_model
from segmenter.augmentations import predict_augments
from segmenter.loss import get_loss
from segmenter.data import DataGenerator, augmented_generator
from segmenter.optimizers import get_optimizer
from segmenter.callbacks import get_evaluation_callbacks
from segmenter.aggregators import Aggregator
import json
import os


def metric_evaluation(clazz: str, job_config, job_hash: str, datadir: str,
                      outdir: str, aggregator: Aggregator):
    K.clear_session()

    _generators, dataset, num_images = augmented_generator(
        clazz, None, predict_augments, job_config, "evaluate", datadir)
    dataset = dataset.batch(1, drop_remainder=True)

    model_dir = os.path.join(outdir, job_hash, clazz, "results",
                             aggregator.name())
    loss = get_loss(job_config["LOSS"])
    print("Creating model for %s" % model_dir)
    model = full_model(clazz,
                       outdir,
                       job_config,
                       job_hash,
                       aggregator=aggregator.layer(),
                       fold_activation=aggregator.fold_activation(),
                       final_activation=aggregator.final_activation())

    os.makedirs(model_dir, exist_ok=True)
    model.save_weights(os.path.join(model_dir, "weights.h5"))

    results = {}
    for threshold in aggregator.thresholds():
        threshold_str = "{:1.2f}".format(threshold)
        threshold_dir = os.path.join(model_dir, threshold_str)
        os.makedirs(threshold_dir, exist_ok=True)

        print("Aggregator {} and Threshold: {}".format(aggregator.name(),
                                                       threshold_str))
        metrics = get_metrics(threshold, job_config["LOSS"])
        model.compile(optimizer=get_optimizer(job_config["OPTIMIZER"]),
                      loss=loss,
                      metrics=list(metrics.values()))
        results[threshold] = model.evaluate(
            x=dataset,
            callbacks=get_evaluation_callbacks(),
            verbose=1,
            steps=num_images)

        with open(os.path.join(threshold_dir, "results.json"),
                  "w") as results_json:
            results_dict = dict(
                zip(['loss'] + list(metrics.keys()),
                    [float(r) for r in results[threshold]]))
            json.dump(results_dict, results_json)
