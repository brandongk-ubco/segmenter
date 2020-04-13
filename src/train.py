from config import get_config

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Average, Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid

import numpy as np

from metrics import get_metrics
from models import get_model, find_latest_weight, find_best_weight
from loss import get_loss
from callbacks import get_callbacks
from optimizers import get_optimizer
from DataGenerator import DataGenerator
from augmentations import train_augments, val_augments
from ops import AverageSingleGradient, AddSingleGradient
import sys

import json
import os
import pprint
import time

from helpers import *

job_config = get_config()
job_hash = hash(job_config)

outdir = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(outdir)
start_time = time.time()

# Turn this on when debugging functions.
if os.environ.get("DEBUG", "false").lower() == "true":
    tf.config.experimental_run_functions_eagerly(True)
else:
    tf.get_logger().setLevel("ERROR")

def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    x = K.clip(x, K.epsilon(), 1 - K.epsilon())
    return - K.log(1. / x - 1.)

def train(clazz, fold=None):
    if job_config["BOOST_FOLDS"] is None:
        train_fold(clazz, fold)
    else:
        for boost_fold in range(0, job_config["BOOST_FOLDS"] + 1):
            train_fold(clazz, fold, boost_fold)

def train_fold(clazz, fold=None, boost_fold=None, activation="sigmoid"):
    K.clear_session()
    K.set_floatx(job_config["PRECISION"])

    print("Training class {}".format(clazz))
    if fold is not None:
        print("Training fold {}".format(fold))

    fold_name = "all" if fold is None else "fold{}".format(fold)
    if boost_fold is not None:
        fold_name += "b{}".format(boost_fold)

    output_folder = os.path.join(outdir, job_hash, clazz, fold_name)
    print("Using directory %s" % output_folder)

    early_stopping_file = os.path.join(output_folder, "early_stopping.json")
    if os.path.isfile(early_stopping_file):
        with open(early_stopping_file, 'r') as stopping_json:
            stopping_state = json.load(stopping_json)
        if stopping_state["stopped_epoch"] > 0:
            print("Fold {} already completed training.".format(fold_name))
            return

    train_generator, train_dataset, num_training_images = generate_for_augments(
        clazz,
        fold,
        train_augments,
        job_config,
        mode="train",
        shuffle=True,
        repeat=True
    )
    image_size = next(train_generator.generate())[0].shape
    train_dataset = train_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=True)

    val_generator, val_dataset, num_val_images = generate_for_augments(
        clazz,
        fold,
        val_augments,
        job_config,
        mode="val",
        repeat=True
    )
    # val_dataset = val_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=True)
    val_dataset = val_dataset.batch(1, drop_remainder=True)

    print("Found %s training images" % num_training_images)
    print("Found %s validation images" % num_val_images)

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(os.path.join(outdir, job_hash), "config.json"), "w") as outfile:
        json.dump(job_config, outfile, indent=4)

    latest_weight = find_latest_weight(output_folder)

    boost_folds = range(0, boost_fold) if boost_fold is not None else []

    threshold = job_config["FSCORE_THRESHOLD"]
    metrics = get_metrics(threshold, job_config["LOSS"])
    metrics = list(metrics.values())
    loss = get_loss(job_config["LOSS"])
    optimizers = get_optimizer(job_config["OPTIMIZER"])

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        boost_models = []

        inputs = Input(shape=image_size)

        for boost_fold in boost_folds:
            boost_fold_model = get_model(image_size, job_config)

            boost_fold_name = "all" if fold is None else "fold{}".format(fold)
            boost_fold_name += "b{}".format(boost_fold)

            boost_fold_dir = os.path.abspath(os.path.join(output_folder, "..", boost_fold_name))
            best_weight = find_best_weight(boost_fold_dir)

            print("Loading weight %s" % best_weight)
            boost_fold_model.load_weights(best_weight)
            boost_fold_model.trainable = False
            boost_fold_model._name = boost_fold_name

            out = boost_fold_model(inputs)
            out = Lambda(lambda x: logit(x), name="{}_logit".format(boost_fold_name))(out)
            boost_models.append(out)

        fold_model = get_model(image_size, job_config)
        fold_model._name = fold_name

        if latest_weight is not None:
            print("Loading weight %s" % latest_weight)
            fold_model.load_weights(latest_weight)

        out = fold_model(inputs)
        if len(boost_models) > 0:
            out = Lambda(lambda x: logit(x), name="{}_logit".format(fold_name))(out)
            out = AddSingleGradient()(boost_models + [out])
            out = Activation(activation, name=activation)(out)
        model = Model(inputs, out)
        model.summary()

        model.compile(
            optimizer=optimizers[0] if isinstance(optimizers, list) else optimizers,
            loss=loss,
            metrics=metrics
        )

    initial_epoch = 0
    val_loss = np.Inf
    if latest_weight is not None:
        initial_epoch = int(latest_weight.split("/")[-1].split("-")[0])
        val_loss = float(latest_weight.split("/")[-1].split("-")[1][:-3])

    train_steps = int(num_training_images/job_config["BATCH_SIZE"])
    val_steps = int(num_val_images/job_config["BATCH_SIZE"])

    history = model.fit(
        initial_epoch=initial_epoch,
        x=train_dataset,
        validation_data=val_dataset,
        epochs=1000,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=get_callbacks(output_folder, job_config, fold, val_loss, start_time, fold_name, loss, metrics, optimizers),
        verbose=1
    )

if __name__ == "__main__":
    pprint.pprint(job_config)

    classes = [os.environ.get("CLASS")] if os.environ.get("CLASS") is not None else job_config["CLASSES"]
    if job_config["FOLDS"] is not None:
        folds = [int(os.environ.get("FOLD"))] if os.environ.get("FOLD") is not None else range(job_config["FOLDS"])
        for clazz in classes:
            for fold in folds:
                train(clazz, fold)
    else:
        for clazz in classes:
            train(clazz)

