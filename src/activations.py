from config import get_config

import pprint

from helpers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from metrics import get_metrics
from models import full_model
from augmentations import predict_augments
from loss import get_loss
from DataGenerator import DataGenerator
from aggregators import get_aggregators
from callbacks import get_evaluation_callbacks
import json
from matplotlib import pyplot as plt
from optimizers import get_optimizer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np

job_config = get_config()
job_hash = hash(job_config)

outdir = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(outdir)

def get_outputs(model, matcher="activation"):
    outputs = []
    for m in [layer for layer in model.layers if isinstance(layer, (Model))]:
        outputs.append((m, [l.output for l in m.layers if matcher in str(type(l)).lower()]))
    return outputs
    

def activations(clazz):
    K.clear_session()

    generators, dataset, num_images = generate_for_augments(clazz, None, predict_augments, job_config, mode="evaluate")
    dataset = dataset.batch(1, drop_remainder=True)

    results = {}
    aggregator_name, aggregator, fold_activation, final_activation, thresholds = get_aggregators(job_config)[0]

    model_dir = os.path.join(outdir, job_hash, clazz, "results", aggregator_name)
    loss = get_loss(job_config["LOSS"])

    print("Creating model for %s" % model_dir)
    model = full_model(clazz, outdir, job_config, job_hash, aggregator=aggregator, fold_activation=fold_activation, final_activation=final_activation)
    os.makedirs(model_dir, exist_ok=True)
    model.save_weights(os.path.join(model_dir, "weights.h5"))

    model.compile(
        optimizer=get_optimizer(job_config["OPTIMIZER"]),
        loss=loss,
        metrics=[]
    )

    K.set_learning_phase(0)
    bins = np.linspace(-10, 10, num=2001)

    for layer_type in ["activation", "convolutional", "normalization"]:
        hist = np.zeros(np.histogram([], bins=bins)[0].shape, dtype="uint64")
        all_outputs = get_outputs(model, matcher=layer_type)
        for i, (m, outputs) in enumerate(all_outputs):
            for batch, (images, masks) in enumerate(dataset):
                print("{} model {} {}/{} image {}/{}".format(layer_type, m.name, i+1, len(all_outputs), batch+1, num_images))
                for o in Model(m.input, outputs).predict_on_batch(images):
                    hist += np.histogram(o, bins=bins)[0].astype(hist.dtype)
        results[layer_type] = hist

    np.savez_compressed(os.path.join(model_dir, activation_histogram), **results)

if __name__ == "__main__":
    pprint.pprint(job_config)

    classes = job_config["CLASSES"] if os.environ.get("CLASS") is None else [os.environ.get("CLASS")]
    print("Visualizing classes %s" % (classes))
    for clazz in classes:
        activations(clazz)
