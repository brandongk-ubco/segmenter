from config import get_config

import pprint

from helpers import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Average
from tensorflow.keras.models import Model

from models import get_model
from augmentations import predict_augments
from loss import NormalizedFocalLoss

from DataGenerator import DataGenerator
from segmentation_models.metrics import FScore

from callbacks import get_evaluation_callbacks
from math import ceil

job_config = get_config()
job_hash = hash(job_config)

def evaluate(clazz):
    pprint.pprint(job_config)

    print("Evaluating class %s" % clazz)

    generators, dataset, num_images = generate_for_augments(clazz, None, predict_augments, job_config, mode="evaluate")
    dataset = dataset.batch(1, drop_remainder=True)
    image_size = next(generators.generate())[0].shape
    print("Found %s images" % num_images)

    inputs = Input(shape=image_size)

    models = []

    for fold in range(job_config["FOLDS"]):
        fold_model = get_model(image_size, job_config)
        best_weight = find_best_weight("/output/%s/%s/fold%s/" % (job_hash, clazz, fold))
        print("Loading weight %s" % best_weight)
        fold_model.load_weights(best_weight)
        fold_model.trainable = False
        fold_model._name = "fold%s" % fold

        out = fold_model(inputs)
        models.append(out)

    out = Average()(models)
    model = Model(inputs, out)
    model.summary()

    model.compile(
        optimizer=Adam(
            learning_rate=job_config["LR"],
            beta_1=job_config["BETA_1"],
            beta_2=job_config["BETA_2"],
            amsgrad=job_config["AMSGRAD"]
        ),
        loss=NormalizedFocalLoss(threshold=job_config["FSCORE_THRESHOLD"]),
        metrics=[FScore(threshold=job_config["FSCORE_THRESHOLD"])]
    )

    history = model.evaluate(
        x=dataset,
        callbacks=get_evaluation_callbacks(),
        verbose=1,
        steps=num_images
    )


if __name__ == "__main__":
    if os.environ.get("CLASS") is not None:
        evaluate(os.environ.get("CLASS"))
    else:
        for clazz in job_config["CLASSES"]:
            evaluate(clazz)
