from config import get_config
import pprint
from helpers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from metrics import get_metrics
from models import full_model
from augmentations import predict_augments
from loss import get_loss
from DataGenerator import DataGenerator
from optimizers import get_optimizer
from callbacks import get_evaluation_callbacks
from aggregators import get_aggregators
import json
import os

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

job_config = get_config()
job_hash = hash(job_config)

outdir = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(outdir) 

def evaluate(clazz, aggregators=None):
    K.clear_session()

    generators, dataset, num_images = generate_for_augments(clazz, None, predict_augments, job_config, mode="evaluate")
    dataset = dataset.batch(1, drop_remainder=True)

    for aggregator_name, aggregator, fold_activation, final_activation, thresholds in get_aggregators(job_config, aggregators):
        model_dir = os.path.join(outdir, job_hash, clazz, "results", aggregator_name)
        loss = get_loss(job_config["LOSS"])
        print("Creating model for %s" % model_dir)
        model = full_model(clazz, outdir, job_config, job_hash, aggregator=aggregator, fold_activation=fold_activation, final_activation=final_activation)

        os.makedirs(model_dir, exist_ok=True)
        model.save_weights(os.path.join(model_dir, "weights.h5"))

        results = {}
        for threshold in thresholds:
            threshold_str = "{:1.2f}".format(threshold)
            threshold_dir = os.path.join(model_dir, threshold_str)
            os.makedirs(threshold_dir, exist_ok=True)
            
            print("Aggregator {} and Threshold: {}".format(aggregator_name, threshold_str))
            metrics = get_metrics(threshold, job_config["LOSS"])
            model.compile(
                optimizer=get_optimizer(job_config["OPTIMIZER"]),
                loss=loss,
                metrics=list(metrics.values())
            )
            results[threshold] = model.evaluate(
                x=dataset,
                callbacks=get_evaluation_callbacks(),
                verbose=1,
                steps=num_images
            )

            with open(os.path.join(threshold_dir, "results.json"), "w") as results_json:
                results_dict = dict(zip(['loss'] + list(metrics.keys()), [float(r) for r in results[threshold]]))
                json.dump(results_dict, results_json)

if __name__ == "__main__":
    pprint.pprint(job_config)

    classes = job_config["CLASSES"] if os.environ.get("CLASS") is None else [os.environ.get("CLASS")]
    aggregators = os.environ.get("AGGREGATORS").split(",") if os.environ.get("AGGREGATORS") is not None else None
    print("Evaluating classes %s" % (classes))
    for clazz in classes:
        evaluate(clazz, aggregators)
