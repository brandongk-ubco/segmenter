import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Average, Lambda, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.activations import sigmoid
from segmenter.metrics import get_metrics
from segmenter.models.full_model import get_model
from segmenter.loss import get_loss
from segmenter.callbacks import get_callbacks
from segmenter.optimizers import get_optimizer
from segmenter.augmentations import train_augments, val_augments
from segmenter.layers import AddSingleGradient
from segmenter.data import augmented_generator
from segmenter.helpers.logit import logit
from segmenter.helpers.parse_fold import parse_fold
from segmenter.models.LatestFoldWeightFinder import LatestFoldWeightFinder
from segmenter.models.BestFoldWeightFinder import BestFoldWeightFinder
import sys
import json
import os
import pprint
import time

start_time = time.time()


def train_fold(clazz,
               fold_name,
               job_config,
               job_hash,
               datadir,
               outdir,
               activation="sigmoid"):
    K.clear_session()
    K.set_floatx(job_config["PRECISION"])

    fold, boost_fold = parse_fold(fold_name)

    print("Training Class {} fold {}".format(clazz, fold_name))

    output_folder = os.path.join(outdir, job_hash, clazz, fold_name)
    print("Using directory %s" % output_folder)

    early_stopping_file = os.path.join(output_folder, "early_stopping.json")
    if os.path.isfile(early_stopping_file):
        with open(early_stopping_file, 'r') as stopping_json:
            stopping_state = json.load(stopping_json)
        if stopping_state["stopped_epoch"] > 0:
            print("Fold {} already completed training.".format(fold_name))
            return

    train_generator, train_dataset, num_training_images = augmented_generator(
        clazz,
        fold,
        train_augments,
        job_config,
        "train",
        datadir,
        shuffle=True,
        repeat=True)
    image_size = next(train_generator.generate())[0].shape
    train_dataset = train_dataset.batch(job_config["BATCH_SIZE"],
                                        drop_remainder=True)

    _val_generator, val_dataset, num_val_images = augmented_generator(
        clazz, fold, val_augments, job_config, "val", datadir, repeat=True)
    val_dataset = val_dataset.batch(job_config["BATCH_SIZE"],
                                    drop_remainder=True)

    print("Found %s training images" % num_training_images)
    print("Found %s validation images" % num_val_images)

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(os.path.join(outdir, job_hash), "config.json"),
              "w") as outfile:
        json.dump(job_config, outfile, indent=4)

    latest_weight_finder = LatestFoldWeightFinder(
        os.path.join(outdir, job_hash, clazz))
    best_weight_finder = BestFoldWeightFinder(
        os.path.join(outdir, job_hash, clazz))

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

            boost_fold_dir = os.path.abspath(
                os.path.join(output_folder, "..", boost_fold_name))
            best_weight = best_weight_finder.get(boost_fold_dir)

            print("Loading weight %s" % best_weight)
            boost_fold_model.load_weights(best_weight)
            boost_fold_model.trainable = False
            boost_fold_model._name = boost_fold_name

            out = boost_fold_model(inputs)
            out = Lambda(lambda x: logit(x),
                         name="{}_logit".format(boost_fold_name))(out)
            boost_models.append(out)

        fold_model = get_model(image_size, job_config)
        fold_model._name = fold_name

        if fold_name in latest_weight_finder.keys():
            latest_weight = latest_weight_finder.get(fold_name)
            print("Loading weight %s" % latest_weight)
            fold_model.load_weights(latest_weight)

        out = fold_model(inputs)
        if len(boost_models) > 0:
            out = Lambda(lambda x: logit(x),
                         name="{}_logit".format(fold_name))(out)
            out = AddSingleGradient()(boost_models + [out])
            out = Activation(activation, name=activation)(out)
        model = Model(inputs, out)
        fold_model.summary()
        model.summary()

        model.compile(optimizer=optimizers[0]
                      if isinstance(optimizers, list) else optimizers,
                      loss=loss,
                      metrics=metrics)

    initial_epoch = 0
    val_loss = np.Inf
    if fold_name in latest_weight_finder.keys():
        latest_weight = latest_weight_finder.get(fold_name)
        initial_epoch = int(latest_weight.split("/")[-1].split("-")[0])
        val_loss = float(latest_weight.split("/")[-1].split("-")[1][:-3])

    train_steps = int(num_training_images / job_config["BATCH_SIZE"])
    val_steps = int(num_val_images / job_config["BATCH_SIZE"])

    model.fit(initial_epoch=initial_epoch,
              x=train_dataset,
              validation_data=val_dataset,
              epochs=1000,
              steps_per_epoch=train_steps,
              validation_steps=val_steps,
              callbacks=get_callbacks(output_folder, job_config, fold,
                                      val_loss, start_time, fold_name, loss,
                                      metrics, optimizers),
              verbose=1)
