
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN, TerminateOnNaN, CSVLogger, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Average, Activation
from tensorflow.keras.models import Model
from segmentation_models.metrics import f1_score, FScore
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy
import numpy as np

from activations import get_activation
from models import unet
from config import get_config
from loss import NormalizedFocalLoss
from callbacks import EarlyStoppingByTime, SavableEarlyStopping, SavableReduceLROnPlateau, AdamSaver, SubModelCheckpoint
from DataGenerator import DataGenerator
from augmentations import train_augment, val_augment
import sys
import hashlib
import json
import os
import psutil
import pprint

from segmentation_models import Unet

def hash(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()

def find_latest_weight(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")]
    if len(files) == 0:
        return None
    return max(files, key=os.path.getctime)

def find_best_weight(folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")]
    if len(files) == 0:
        return None
    return min(files, key=lambda x: float(x.split("-")[1]))

def get_callbacks(output_folder, job_config, fold, val_loss, train_generator):

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

    model_autosave = SubModelCheckpoint(
        filepath=os.path.join(output_folder, "{epoch:04d}-{val_loss:.4f}-{val_f1-score:.4f}.h5"),
        submodel="fold%s" % fold,
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

    return [optimizer_saver, lr_reducer, TerminateOnNaN(), early_stopping, logger, tensorboard, model_autosave, time_limit, train_shuffler]

def make_shape(image, mask):
    image.set_shape([None, None, None])
    mask.set_shape([None, None, None])
    return image, mask

# Do two calls per CPU
parallel_data_calls= 2 * max(psutil.cpu_count(logical=False) - 1, 1)

def get_model(image_size, job_config):

    regularizer = l1_l2(l1=job_config["L1_REG"], l2=job_config["L2_REG"])

    model = unet(
        input_shape=image_size,
        use_batch_norm=job_config["BATCH_NORM"],
        filters=job_config["FILTERS"],
        dropout=job_config["DROPOUT"],
        dropout_change_per_layer=job_config["DROPOUT_CHANGE_PER_LAYER"],
        use_dropout_on_upsampling=job_config["USE_DROPOUT_ON_UPSAMPLE"],
        activation=get_activation(job_config["ACTIVATION"]),
        kernel_initializer='he_normal',
        num_layers=job_config["LAYERS"]
    )

    # base_model = Unet(
    #     backbone_name='efficientnetb0'
    # )
    # inp = Input(shape=(None, None, 1))
    # l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
    # out = base_model(l1)
    # model = Model(inp, out, name=base_model.name)

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    return model

job_config = get_config()

def train_fold(clazz, fold):
            pprint.pprint(job_config)

            print("Training class %s, fold %s of %s" % (clazz, fold, job_config["NUM_FOLDS"]))

            train_generator = DataGenerator(clazz, fold, augmentations=train_augment, job_config=job_config)
            train_dataset = tf.data.Dataset.from_generator(train_generator.generate, (tf.float32,tf.float32))
            train_dataset = train_dataset.map(
                lambda image, mask: tf.numpy_function(train_generator.augment, [image, mask], [tf.float32, tf.float32]),
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

            val_generator = DataGenerator(clazz, fold, mode="val", job_config=job_config)
            val_dataset = tf.data.Dataset.from_generator(val_generator.generate, (tf.float32,tf.float32))
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

            latest_weight = find_latest_weight(output_folder)
            backend.clear_session()

            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():

                previous_models = []

                inputs = Input(shape=image_size)

                for previous_fold in range(max(0, fold - job_config["TRAIN_BEHIND"]), fold):
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
                if fold > 0:
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
                callbacks=get_callbacks(output_folder, job_config, fold, val_loss, train_generator),
                verbose=1
            )

if __name__ == "__main__":
    if os.environ.get("TRAIN_CLASS") is not None:
        for fold in range(job_config["NUM_FOLDS"]):
            train_fold(os.environ.get("TRAIN_CLASS"), fold)
    else:
        for clazz in job_config["CLASSES"]:
            for fold in range(job_config["NUM_FOLDS"]):
                train_fold(clazz, fold)
