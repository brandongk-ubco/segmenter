from config import get_config

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow.keras.layers import Input, Average
from tensorflow.keras.models import Model

from segmentation_models.metrics import FScore
import numpy as np

from models import get_model
from loss import NormalizedFocalLoss
from callbacks import get_callbacks
from DataGenerator import DataGenerator
from augmentations import train_augments, val_augments
import sys

import json
import os
import pprint

from helpers import *

job_config = get_config()
job_hash = hash(job_config)

def train_fold(clazz, fold):
            pprint.pprint(job_config)

            print("Training class %s, fold %s (NUM_FOLDS = %s)" % (clazz, fold, job_config["FOLDS"]))

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

            output_folder = "/output/%s/%s/fold%s/" % (job_hash, clazz, fold)
            os.makedirs(output_folder, exist_ok=True)

            with open(os.path.join("/output/%s" % job_hash, "config.json"), "w") as outfile:
                json.dump(job_config, outfile, indent=4)

            latest_weight = find_latest_weight(output_folder)
            backend.clear_session()

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():

                previous_models = []

                inputs = Input(shape=image_size)

                train_behind_start = max(0, fold - job_config["TRAIN_BEHIND"])

                for previous_fold in range(train_behind_start, fold):
                    previous_fold_model = get_model(image_size, job_config)
                    best_weight = find_best_weight("/output/%s/%s/fold%s/" % (job_hash, clazz, previous_fold))
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
                    out = Average()(previous_models + [out])
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
                callbacks=get_callbacks(output_folder, job_config, fold, val_loss),
                verbose=1
            )

if __name__ == "__main__":
    if os.environ.get("CLASS") is not None and os.environ.get("FOLD") is not None:
        train_fold(os.environ.get("CLASS"), int(os.environ.get("FOLD")))
    elif os.environ.get("CLASS") is not None:
        for fold in range(job_config["FOLDS"]):
            train_fold(os.environ.get("CLASS"), fold)
    else:
        for clazz in job_config["CLASSES"]:
            for fold in range(job_config["FOLDS"]):
                train_fold(clazz, fold)
