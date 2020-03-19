from config import get_config

import pprint

from helpers import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from metrics import get_metrics

from models import model_for_folds
from augmentations import predict_augments
from loss import get_loss

from DataGenerator import DataGenerator

from callbacks import get_evaluation_callbacks
import json

job_config = get_config()
job_hash = hash(job_config)

outdir = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(outdir)

def evaluate(clazz, folds=None):
    K.clear_session()

    generators, dataset, num_images = generate_for_augments(clazz, None, predict_augments, job_config, mode="evaluate")
    dataset = dataset.batch(1, drop_remainder=True)

    if folds is None:
        folds = range(job_config["FOLDS"])

    model_dir = os.path.join(outdir, job_hash, clazz, "results", "-".join([str("fold%s" % f) for f in folds]))

    loss = get_loss(job_config["LOSS"])

    print("Creating model for %s" % model_dir)
    model = model_for_folds(clazz, outdir, job_config, job_hash, folds=folds)

    os.makedirs(model_dir, exist_ok=True)
    model.save_weights(os.path.join(model_dir, "weights.h5"))

    results = {}
    for threshold in np.linspace(0.05, 0.95, num=19):
        metrics = get_metrics(threshold, job_config["LOSS"])
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
        results[threshold] = model.evaluate(
            x=dataset,
            callbacks=get_evaluation_callbacks(),
            verbose=1,
            steps=num_images
        )

        with open(os.path.join(model_dir, "%0.2f-threshold-results.json" % threshold), "w") as results_json:
            results_dict = dict(zip(['loss'] + list(metrics.keys()), [float(r) for r in results[threshold]]))
            json.dump(results_dict, results_json)

    return results

if __name__ == "__main__":
    pprint.pprint(job_config)

    classes = job_config["CLASSES"] if os.environ.get("CLASS") is None else [os.environ.get("CLASS")]

    if os.environ.get("EVAL_FOLDS") is not None:
        folds = [int(f.strip()) for f in os.environ.get("EVAL_FOLDS").split(",")]
    elif os.environ.get("FOLD") is not None:
        folds = [int(os.environ.get("FOLD").strip())]
    else:
        folds = range(job_config["FOLDS"])

    folds = list(folds)
    print("Evaluating classes %s and folds %s" % (classes, folds))
    evaluations = {}
    for clazz in classes:
        evaluations[clazz] = evaluate(clazz, folds=folds)
    pprint.pprint(evaluations)
