from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard, TerminateOnNaN, ProgbarLogger, BaseLogger, TerminateOnNaN, CSVLogger, LambdaCallback
from segmentation_models import Unet
from tensorflow.keras.optimizers import Adam
from segmentation_models.metrics import f1_score
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy, DiceLoss, BinaryFocalLoss
from segmentation_models.base import Loss
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
from models import unet
import os
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend
import tensorflow as tf
from math import ceil
from data import DataGenerator
from augmentations import augment
import time
import sys
from config import get_config
import hashlib
import json

class NormalizedFocalLoss(Loss):

    def __call__(self, gt, pr, **kwargs):
        dl = dice_loss(gt, pr, **kwargs)
        bfl = BinaryFocalLoss(alpha=1, gamma=5)(gt, pr, **kwargs)
        bce = binary_crossentropy(gt, pr, **kwargs)

        ce_loss = bfl + bce
        
        return ce_loss + ce_loss * dl / backend.mean(ce_loss)

class EarlyStoppingByTime(Callback):
    def __init__(self, limit_seconds, verbose=0):
        super(Callback, self).__init__()
        self.start = time.time()
        self.limit_seconds = limit_seconds
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = time.time()
        elapsed = current - self.start

        if elapsed > self.limit_seconds:
            print("Epoch %05d: Time limit exhausted (%s seconds)" % (epoch+1, self.limit_seconds))
            sys.exit(123)

        if self.verbose > 0:
            print("Epoch %05d: %.2f seconds remaining" % (epoch, self.limit_seconds - elapsed))

def hash(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()

def find_best_weight(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and f != "logs"]
    if len(files) == 0:
        return None
    return max(files, key=os.path.getctime)

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

            image_size = next(val_generator.generate())[0].shape
            print(image_size)

            train_dataset_hash = hash(json.dumps(sorted(train_generator.fold_data)))
            val_dataset_hash = hash(json.dumps(sorted(val_generator.fold_data)))
            job_config_hash = hash(job_config)
            job_hash = hash(train_dataset_hash + val_dataset_hash + job_config_hash)

            output_folder = "/output/%s/%s/fold%s/" % (job_hash, clazz, fold)
            os.makedirs(output_folder, exist_ok=True)

            with open(os.path.join("/output/%s" % job_hash, "config.json"), "w") as outfile:
                json.dump(job_config, outfile, indent=4)

            log_folder = os.path.join(output_folder, "logs")
            os.makedirs(log_folder, exist_ok=True)

            best_weight = find_best_weight(output_folder)

            print("Found %s training images" % num_training_images)
            print("Found %s validation images" % num_val_images)

            lr_reducer = ReduceLROnPlateau(factor=job_config["LR_REDUCTION_FACTOR"],
                                        cooldown=job_config["PATIENCE"],
                                        patience=job_config["PATIENCE"], verbose=1,
                                        min_lr=job_config["MIN_LR"],
                                        monitor='val_loss',
                                        mode='min',
                                        min_delta=1e-2)

            mode_autosave = ModelCheckpoint(os.path.join(output_folder, "{epoch:04d}-{val_loss:.4f}-{val_f1-score:.4f}"), monitor='val_loss',
                                            mode='min', save_best_only=True, verbose=1)

            early_stopping = EarlyStopping(patience=job_config["PATIENCE"]*3, verbose=1, monitor='val_loss', mode='min')

            tensorboard = TensorBoard(log_dir=os.path.join(log_folder, "tenboard"), profile_batch=0)

            logger = CSVLogger(os.path.join(log_folder, 'train.csv'), separator=',', append=True)

            train_shuffler = LambdaCallback(on_epoch_end= lambda epoch, logs: train_generator.shuffle())

            time_limit = EarlyStoppingByTime(limit_seconds=os.environ.get("LIMIT_SECONDS", 60 * 60), verbose=1)
            
            callbacks = [mode_autosave, TerminateOnNaN(), time_limit, early_stopping, lr_reducer, logger, tensorboard, train_shuffler]

            initial_epoch = 0
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():

                if best_weight is not None:
                    print("Resuming training from %s" % best_weight)
                    model = tf.keras.models.load_model(best_weight, custom_objects={'NormalizedFocalLoss': NormalizedFocalLoss(), 'f1-score': f1_score})
                    initial_epoch = int(best_weight.split("/")[-1].split("-")[0])
                else:
                    model = unet(input_shape=image_size, use_batch_norm=job_config["BATCH_NORM"], filters=job_config["FILTERS"], dropout=job_config["DROPOUT"], dropout_change_per_layer=0, use_dropout_on_upsampling=True)

                    regularizer = l1_l2(l1=job_config["L1_REG"], l2=job_config["L2_REG"])

                    for layer in model.layers:
                        for attr in ['kernel_regularizer']:
                            if hasattr(layer, attr):
                                setattr(layer, attr, regularizer)

                    model.compile(optimizer=Adam(learning_rate=job_config["LR"], beta_1=job_config["BETA_1"], beta_2=job_config["BETA_2"], amsgrad=job_config["AMSGRAD"]),
                            loss=NormalizedFocalLoss(), metrics=[f1_score])

            train_steps = int(num_training_images/job_config["BATCH_SIZE"])
            val_steps = int(num_val_images/job_config["BATCH_SIZE"])

            history = model.fit(initial_epoch=initial_epoch, x=train_dataset, validation_data=val_dataset,
                                        epochs=1000, steps_per_epoch=train_steps,
                                        validation_steps=val_steps, callbacks=callbacks, verbose=1)

if __name__ == "__main__":
    if os.environ.get("TRAIN_CLASS") is not None and os.environ.get("TRAIN_FOLD") is not None:
        train_fold(os.environ.get("TRAIN_CLASS"), int(os.environ.get("TRAIN_FOLD")))
    else:
        for clazz in ["1", "2", "3", "4"]:
            for fold in range(5):
                train_fold(clazz, fold)