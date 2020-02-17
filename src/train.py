
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN, TerminateOnNaN, CSVLogger, LambdaCallback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal, RandomUniform
from tensorflow.keras.layers import Input, Conv2D, Concatenate, Average, Activation
from tensorflow.keras.models import Model
from segmentation_models.metrics import f1_score, FScore
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy
import numpy as np

from models import get_model
from config import get_config
from loss import NormalizedFocalLoss
from callbacks import EarlyStoppingByTime, SavableEarlyStopping, SavableReduceLROnPlateau, AdamSaver, SubModelCheckpoint
from DataGenerator import DataGenerator
from augmentations import train_augments, val_augments
import sys
import hashlib
import json
import os
import psutil
import pprint

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

def get_callbacks(output_folder, job_config, fold, val_loss, train_generators):

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

    train_shuffler = LambdaCallback(on_epoch_end= lambda epoch, logs: [g.shuffle() for g in train_generators])

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

job_config = get_config()

def generator_to_dataset(generator):
    dataset = tf.data.Dataset.from_generator(generator.generate, (tf.float32,tf.float32))
    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(generator.augment, [image, mask], [tf.float32, tf.float32]),
        num_parallel_calls=parallel_data_calls
    )
    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(generator.recenter, [image, mask], [tf.float16, tf.float16]),
        num_parallel_calls=parallel_data_calls
    )
    dataset = dataset.map(
        lambda image, mask: make_shape(image, mask),
        num_parallel_calls=parallel_data_calls
    )
    return dataset

def generate_for_all_augments(augments, mode):
    combined_dataset = None
    generators = []
    for augment in augments():
        generator = DataGenerator(clazz, fold, augmentations=augment, job_config=job_config, mode=mode)
        generators.append(generator)
        dataset = generator_to_dataset(generator)
        if combined_dataset is None:
            combined_dataset = dataset
        else:
            combined_dataset.concatenate(dataset)
    return generators, combined_dataset

def train_fold(clazz, fold):
            pprint.pprint(job_config)

            print("Training class %s, fold %s of %s" % (clazz, fold, job_config["NUM_FOLDS"]))

            train_generators, train_dataset = generate_for_all_augments(train_augments, mode="train")
            image_size = next(train_generators[0].generate())[0].shape
            num_training_images = sum([g.size() for g in train_generators])
            train_dataset = train_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=False)

            val_generators, val_dataset = generate_for_all_augments(val_augments, mode="val")
            num_val_images = sum([g.size() for g in val_generators])
            val_dataset = val_dataset.batch(job_config["BATCH_SIZE"], drop_remainder=False)

            print("Found %s training images" % num_training_images)
            print("Found %s validation images" % num_val_images)

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
                callbacks=get_callbacks(output_folder, job_config, fold, val_loss, train_generators),
                verbose=1
            )

if __name__ == "__main__":
    if os.environ.get("TRAIN_CLASS") is not None and os.environ.get("TRAIN_FOLD") is not None:
        train_fold(os.environ.get("TRAIN_CLASS"), int(os.environ.get("TRAIN_FOLD")))
    elif os.environ.get("TRAIN_CLASS") is not None:
        for fold in range(job_config["NUM_FOLDS"]):
            train_fold(os.environ.get("TRAIN_CLASS"), fold)
    else:
        for clazz in job_config["CLASSES"]:
            for fold in range(job_config["NUM_FOLDS"]):
                train_fold(clazz, fold)
