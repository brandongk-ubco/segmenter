from config import get_config

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Average
from tensorflow.keras.models import Model

import numpy as np

from metrics import get_metrics
from models import get_model, find_latest_weight, find_best_weight
from loss import get_loss
from callbacks import get_callbacks
from DataGenerator import DataGenerator
from augmentations import train_augments, val_augments
from ops import AverageSingleGradient
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

def train_fold(clazz, fold):
    K.clear_session()
    K.set_floatx(job_config["PRECISION"])

    print("Training class %s, fold %s (%s folds)" % (clazz, fold, job_config["FOLDS"]))

    output_folder = os.path.join(outdir, job_hash, clazz, "fold%s" % fold)
    print("Using directory %s" % output_folder)

    if job_config["BOOST_FOLDS"] > 0:
        train_generator, train_dataset, num_training_images = generate_for_augments(
            clazz,
            None,
            train_augments,
            job_config,
            mode="train",
            shuffle=True,
            repeat=True
        )
        image_size = next(train_generator.generate())[0].shape
        train_dataset = train_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=True)

        val_generator, val_dataset, num_val_images = generate_for_augments(clazz, None, val_augments, job_config, mode="val", repeat=True)
        val_dataset = val_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=True)
    else:
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

        val_generator, val_dataset, num_val_images = generate_for_augments(clazz, fold, val_augments, job_config, mode="val", repeat=True)
        val_dataset = val_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=True)

    print("Found %s training images" % num_training_images)
    print("Found %s validation images" % num_val_images)

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(os.path.join(outdir, job_hash), "config.json"), "w") as outfile:
        json.dump(job_config, outfile, indent=4)

    latest_weight = find_latest_weight(output_folder)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():

        previous_models = []

        inputs = Input(shape=image_size)

        boost_fold_start = max(0, fold - job_config["BOOST_FOLDS"])

        for previous_fold in range(boost_fold_start, fold):
            previous_fold_model = get_model(image_size, job_config)
            previous_fold_dir = os.path.abspath(os.path.join(output_folder, "..", "fold%s" % previous_fold))
            best_weight = find_best_weight(previous_fold_dir)
            print("Loading weight %s" % best_weight)
            previous_fold_model.load_weights(best_weight)
            previous_fold_model.trainable = False
            previous_fold_model._name = "fold%s" % previous_fold

            out = previous_fold_model(inputs)
            previous_models.append(out)

        fold_model = get_model(image_size, job_config)
        fold_model._name = "fold%s" % fold

        if latest_weight is not None:
            print("Loading weight %s" % latest_weight)
            fold_model.load_weights(latest_weight)

        out = fold_model(inputs)
        if len(previous_models) > 0:
            out = AverageSingleGradient()(previous_models + [out])
        model = Model(inputs, out)
        model.summary()

        model_memory_usage = get_model_memory_usage(job_config["BATCH_SIZE"], model)
        print("Estimated Training GPU Memory Usage: {:.2f} Gb".format(model_memory_usage))

        threshold = job_config["FSCORE_THRESHOLD"]
        metrics = get_metrics(threshold, job_config["LOSS"])

        model.compile(
            optimizer=Adam(
                learning_rate=job_config["LR"],
                beta_1=job_config["BETA_1"],
                beta_2=job_config["BETA_2"],
                amsgrad=job_config["AMSGRAD"]
            ),
            loss=get_loss(job_config["LOSS"]),
            metrics=list(metrics.values())
        )

    initial_epoch = 0
    val_loss = np.Inf
    if latest_weight is not None:
        initial_epoch = int(latest_weight.split("/")[-1].split("-")[0])
        val_loss = float(latest_weight.split("/")[-1].split("-")[1])

    train_steps = int(num_training_images/job_config["BATCH_SIZE"])
    val_steps = int(num_val_images/job_config["BATCH_SIZE"])

    history = model.fit(
        initial_epoch=initial_epoch,
        x=train_dataset,
        validation_data=val_dataset,
        epochs=1000,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=get_callbacks(output_folder, job_config, fold, val_loss, start_time),
        verbose=1
    )

if __name__ == "__main__":
    pprint.pprint(job_config)
    if os.environ.get("CLASS") is not None and os.environ.get("FOLD") is not None:
        train_fold(os.environ.get("CLASS"), int(os.environ.get("FOLD")))
    elif os.environ.get("CLASS") is not None:
        for fold in range(job_config["FOLDS"]):
            train_fold(os.environ.get("CLASS"), fold)
    else:
        for clazz in job_config["CLASSES"]:
            for fold in range(job_config["FOLDS"]):
                train_fold(clazz, fold)
