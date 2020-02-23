from config import get_config

import pprint

from helpers import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from segmentation_models.metrics import FScore, Precision, Recall
from metrics import Specificity, FallOut

from models import model_for_folds
from augmentations import predict_augments
from loss import NormalizedFocalLoss

from DataGenerator import DataGenerator
from segmentation_models.metrics import FScore

from callbacks import get_evaluation_callbacks
from math import ceil
import json

job_config = get_config()
job_hash = hash(job_config)

outdir = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(outdir)

def evaluate(clazz, folds=None, method="include", overwrite=False):
    K.clear_session()

    generators, dataset, num_images = generate_for_augments(clazz, None, predict_augments, job_config, method=method, mode="evaluate")
    dataset = dataset.batch(1, drop_remainder=True)

    if folds is None:
        folds = range(job_config["FOLDS"])

    model_dir = os.path.join(outdir, job_hash, clazz, "results", "-".join([str("fold%s" % f) for f in folds]))

    threshold = job_config["FSCORE_THRESHOLD"]

    loss = NormalizedFocalLoss(threshold=threshold)
    metrics = {
        "f1-score": FScore(threshold=threshold),
        "precision": Precision(threshold=threshold),
        "recall": Recall(threshold=threshold),
        "specificity": Specificity(threshold=threshold)
    }

    if not os.path.isdir(model_dir) or overwrite:
        print("Creating new model for %s" % model_dir)
        model = model_for_folds(clazz, outdir, job_config, job_hash, folds=folds)
    else:
        print("Loading existing model from %s" % model_dir)
        model = model_for_folds(clazz, outdir, job_config, job_hash, folds=folds, load_weights=False)
        model.load_weights(os.path.join(model_dir, "weights.h5"))

    model.compile(
        optimizer=Adam(
            learning_rate=job_config["LR"],
            beta_1=job_config["BETA_1"],
            beta_2=job_config["BETA_2"],
            amsgrad=job_config["AMSGRAD"]
        ),
        loss=loss,
        metrics=list(metrics.values())
    )

    if not os.path.isdir(model_dir) or overwrite:
        os.makedirs(model_dir, exist_ok=True)
        model.save_weights(os.path.join(model_dir, "weights.h5"))

    results = model.evaluate(
        x=dataset,
        callbacks=get_evaluation_callbacks(),
        verbose=1,
        steps=num_images
    )

    eval_type = "out_of_class" if method == "exclude" else "in_class"
    with open(os.path.join(model_dir, "%s_results.json" % eval_type), "w") as results_json:
        results_dict = dict(zip(['loss'] + list(metrics.keys()), [float(r) for r in results]))
        json.dump(results_dict, results_json)

    return results

if __name__ == "__main__":
    classes = job_config["CLASSES"] if os.environ.get("CLASS") is None else [os.environ.get("CLASS")]
    folds = range(job_config["FOLDS"]) if os.environ.get("EVAL_FOLDS") is None else [int(f.strip()) for f in os.environ.get("EVAL_FOLDS").split(",")]
    print(folds)
    evaluations = {}
    for clazz in classes:
        evaluations[clazz] = {}
        print("In-Class evaluation for %s" % clazz)
        evaluations[clazz]["in_class"] = evaluate(clazz, folds=folds, method="include")
        print("Out-Of-Class evaluation for %s" % clazz)
        evaluations[clazz]["out_of_class"] = evaluate(clazz, folds=folds, method="exclude")
    pprint.pprint(evaluations)
