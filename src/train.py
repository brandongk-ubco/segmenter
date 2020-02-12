
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, TerminateOnNaN, TerminateOnNaN, CSVLogger, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from segmentation_models.metrics import f1_score
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy
import numpy as np

from activations import get_activation
from models import unet
from config import get_config
from loss import NormalizedFocalLoss
from callbacks import EarlyStoppingByTime, SavableEarlyStopping, SavableReduceLROnPlateau, AdamSaver
from DataGenerator import DataGenerator
from augmentations import train_augment, val_augment
import sys
import hashlib
import json
import os
import psutil
import pprint

def hash(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()

def find_best_weight(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")]
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
        verbose=2
    )

    model_autosave = ModelCheckpoint(
        os.path.join(output_folder, "{epoch:04d}-{val_loss:.4f}-{val_f1-score:.4f}.h5"),
        save_best_only=False,
        save_weights_only=True
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
        limit_seconds=int(os.environ.get("LIMIT_SECONDS", -1)),
        verbose=0
    )

    optimizer_saver = AdamSaver(
        os.path.join(output_folder, "optimizer.pkl")
    )

    return [optimizer_saver, lr_reducer, model_autosave, TerminateOnNaN(), early_stopping, logger, tensorboard, train_shuffler, time_limit]

def make_shape(image, mask):
    image.set_shape([None, None, None])
    mask.set_shape([None, None, None])
    return image, mask

# Do two calls per CPU
parallel_data_calls= 2 * max(psutil.cpu_count(logical=False) - 1, 1)

def train_fold(clazz, fold):
            job_config = get_config()

            pprint.pprint(job_config)

            print("Training class %s, fold %s" % (clazz, fold))

            train_folder = '/data/%s/fold%s/train/' % (clazz, fold)
            val_folder = '/data/%s/fold%s/val/' % (clazz, fold)

            train_generator = DataGenerator(clazz, fold, augmentations=train_augment, job_config=job_config)
            train_dataset = tf.data.Dataset.from_generator(train_generator.generate, (tf.float16,tf.float16))
            train_dataset = train_dataset.map(
                lambda image, mask: tf.numpy_function(train_generator.augment, [image, mask], [tf.float16, tf.float16]),
                num_parallel_calls=parallel_data_calls
            )
            train_dataset = train_dataset.map(
                lambda image, mask: tf.numpy_function(train_generator.recenter, [image, mask], [tf.float16, tf.float16]),
                num_parallel_calls=parallel_data_calls
            )
            train_dataset = train_dataset.map(
                lambda image, mask: make_shape(image, mask),
                num_parallel_calls=parallel_data_calls
            )
            train_dataset = train_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=False)

            val_generator = DataGenerator(clazz, fold, mode="val")
            val_dataset = tf.data.Dataset.from_generator(val_generator.generate, (tf.float16,tf.float16))
            val_dataset = val_dataset.map(
                lambda image, mask: tf.numpy_function(train_generator.recenter, [image, mask], [tf.float16, tf.float16]),
                num_parallel_calls=parallel_data_calls
            )
            val_dataset = val_dataset.map(
                lambda image, mask: make_shape(image, mask),
                num_parallel_calls=parallel_data_calls
            )
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

            backend.clear_session()

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():

                model = unet(
                    input_shape=image_size,
                    use_batch_norm=job_config["BATCH_NORM"],
                    filters=job_config["FILTERS"],
                    dropout=job_config["DROPOUT"],
                    dropout_change_per_layer=job_config["DROPOUT_CHANGE_PER_LAYER"],
                    use_dropout_on_upsampling=job_config["USE_DROPOUT_ON_UPSAMPLE"],
                    activation=get_activation(job_config["ACTIVATION"]),
                    kernel_initializer='he_normal',
                    layers=job_config["LAYERS"]
                )

                regularizer = l1_l2(l1=job_config["L1_REG"], l2=job_config["L2_REG"])

                for layer in model.layers:
                    for attr in ['kernel_regularizer']:
                        if hasattr(layer, attr):
                            setattr(layer, attr, regularizer)

                model.compile(
                    optimizer=Adam(
                        learning_rate=job_config["LR"],
                        beta_1=job_config["BETA_1"],
                        beta_2=job_config["BETA_2"],
                        amsgrad=job_config["AMSGRAD"]
                    ),
                    loss=NormalizedFocalLoss(),
                    metrics=[f1_score]
                )

                if best_weight is not None:
                    model.load_weights(best_weight)

            initial_epoch = 0
            val_loss = np.Inf
            if best_weight is not None:
                initial_epoch = int(best_weight.split("/")[-1].split("-")[0])
                val_loss = float(best_weight.split("/")[-1].split("-")[1])

            train_steps = int(num_training_images/job_config["BATCH_SIZE"])
            val_steps = int(num_val_images/job_config["BATCH_SIZE"])

            history = model.fit(
                initial_epoch=initial_epoch,
                x=train_dataset,
                validation_data=val_dataset,
                epochs=1000,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                callbacks=get_callbacks(output_folder, job_config, val_loss, train_generator),
                verbose=1
            )

if __name__ == "__main__":
    if os.environ.get("TRAIN_CLASS") is not None and os.environ.get("TRAIN_FOLD") is not None:
        train_fold(os.environ.get("TRAIN_CLASS"), int(os.environ.get("TRAIN_FOLD")))
    else:
        for clazz in ["1", "2", "3", "4"]:
            train_fold(clazz, 0)