
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN, TerminateOnNaN, CSVLogger, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend
from segmentation_models.metrics import f1_score
import numpy as np

from activations import CosActivation
from models import unet
from config import get_config
from loss import NormalizedFocalLoss
from callbacks import EarlyStoppingByTime, SavableEarlyStopping, SavableReduceLROnPlateau
from DataGenerator import DataGenerator
from augmentations import augment

import sys
import hashlib
import json
import os

def hash(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()

def find_best_weight(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f != "logs"]
    if len(files) == 0:
        return None
    return max(files, key=os.path.getctime)

def get_callbacks(output_folder, job_config, val_loss, train_generator):

    log_folder = os.path.join(output_folder, "logs")
    os.makedirs(log_folder, exist_ok=True)

    lr_reducer = SavableReduceLROnPlateau(
        os.path.join(output_folder, "lr_reducer.json"),
        factor=job_config["LR_REDUCTION_FACTOR"],
        cooldown=job_config["PATIENCE"],
        patience=job_config["PATIENCE"],
        min_lr=job_config["MIN_LR"],
        monitor='val_loss',
        mode='min',
        min_delta=1e-2,
        verbose=2
    )

    model_autosave = ModelCheckpoint(
        os.path.join(output_folder, "{epoch:04d}-{val_loss:.4f}-{val_f1-score:.4f}"),
        save_best_only=False
    )
    model_autosave.best = val_loss

    early_stopping = SavableEarlyStopping(
        os.path.join(output_folder, "early_stopping.json"),
        patience=job_config["PATIENCE"]*3,
        verbose=2,
        monitor='val_loss',
        mode='min'
    )

    tensorboard = TensorBoard(
        log_dir=os.path.join(log_folder, "tenboard"),
        histogram_freq=1,
        embeddings_freq=1,
        profile_batch=0
    )

    logger = CSVLogger(
        os.path.join(log_folder, 'train.csv'),
        separator=',',
        append=True
    )

    train_shuffler = LambdaCallback(on_epoch_end= lambda epoch, logs: train_generator.shuffle())

    time_limit = EarlyStoppingByTime(
        limit_seconds=int(os.environ.get("LIMIT_SECONDS", 60 * 60)),
        verbose=0
    )

    return [lr_reducer, model_autosave, TerminateOnNaN(), early_stopping, logger, tensorboard, train_shuffler, time_limit]
     

def train_fold(clazz, fold):
            job_config = get_config()

            print("Using BATCH_SIZE: %s" % job_config["BATCH_SIZE"])
            print("Using PATIENCE: %s" % job_config["PATIENCE"])

            print("Training class %s, fold %s" % (clazz, fold))

            train_folder = '/data/%s/fold%s/train/' % (clazz, fold)
            val_folder = '/data/%s/fold%s/val/' % (clazz, fold)

            train_generator = DataGenerator(clazz, fold, augmentations=augment)
            train_dataset = tf.data.Dataset.from_generator(train_generator.generate, (tf.float32,tf.float32),
                output_shapes=(tf.TensorShape((None, None, None)), tf.TensorShape((None, None, None))))
            train_dataset = train_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=False)

            val_generator = DataGenerator(clazz, fold, mode="val")
            val_dataset = tf.data.Dataset.from_generator(val_generator.generate, (tf.float32,tf.float32),
                output_shapes=(tf.TensorShape((None, None, None)), tf.TensorShape((None, None, None))))
            val_dataset = val_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=False)

            num_training_images = train_generator.size()
            num_val_images = val_generator.size()

            print("Found %s training images" % num_training_images)
            print("Found %s validation images" % num_val_images)

            image_size = next(val_generator.generate())[0].shape
            job_hash = hash(job_config)

            output_folder = "/output/%s/%s/fold%s/" % (job_hash, clazz, fold)
            os.makedirs(output_folder, exist_ok=True)

            with open(os.path.join("/output/%s" % job_hash, "config.json"), "w") as outfile:
                json.dump(job_config, outfile, indent=4)

            best_weight = find_best_weight(output_folder)

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():

                if best_weight is not None:
                    print("Resuming training from %s" % best_weight)
                    model = tf.keras.models.load_model(best_weight, custom_objects={'NormalizedFocalLoss': NormalizedFocalLoss(), 'f1-score': f1_score})
                else:
                    model = unet(
                        input_shape=image_size,
                        use_batch_norm=job_config["BATCH_NORM"],
                        filters=job_config["FILTERS"],
                        dropout=job_config["DROPOUT"],
                        dropout_change_per_layer=0,
                        use_dropout_on_upsampling=True,
                        activation=CosActivation
                    )

                    regularizer = l1_l2(l1=job_config["L1_REG"], l2=job_config["L2_REG"])

                    for layer in model.layers:
                        for attr in ['kernel_regularizer']:
                            if hasattr(layer, attr):
                                setattr(layer, attr, regularizer)

                    model.compile(optimizer=Adam(learning_rate=job_config["LR"], beta_1=job_config["BETA_1"], beta_2=job_config["BETA_2"], amsgrad=job_config["AMSGRAD"]),
                            loss=NormalizedFocalLoss(), metrics=[f1_score])

            initial_epoch = 0
            val_loss = np.Inf
            if best_weight is not None:
                initial_epoch = int(best_weight.split("/")[-1].split("-")[0])
                val_loss = float(best_weight.split("/")[-1].split("-")[1])

            train_steps = int(num_training_images/job_config["BATCH_SIZE"])
            val_steps = int(num_val_images/job_config["BATCH_SIZE"])

            history = model.fit(initial_epoch=initial_epoch, x=train_dataset, validation_data=val_dataset,
                                        epochs=1000, steps_per_epoch=train_steps,
                                        validation_steps=val_steps, callbacks=get_callbacks(output_folder, job_config, val_loss, train_generator), verbose=1)

if __name__ == "__main__":
    if os.environ.get("TRAIN_CLASS") is not None and os.environ.get("TRAIN_FOLD") is not None:
        train_fold(os.environ.get("TRAIN_CLASS"), int(os.environ.get("TRAIN_FOLD")))
    else:
        for clazz in ["1", "2", "3", "4"]:
            for fold in range(5):
                train_fold(clazz, fold)