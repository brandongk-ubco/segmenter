import hashlib
import os
import tensorflow as tf
from segmenter.DataGenerator import DataGenerator
import numpy as np
from tensorflow.keras import backend as K


def hash(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()


def make_shape(image, mask):
    image.set_shape([None, None, None])
    mask.set_shape([None, None, None])
    return image, mask


def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    x = K.clip(x, K.epsilon(), 1 - K.epsilon())
    return - K.log(1. / x - 1.)


def generator_to_dataset(generator, repeat, shuffle, zero=False, buffer_size=500):
    dataset = tf.data.Dataset.from_generator(
        generator.generate, (K.floatx(), K.floatx()))

    if shuffle:
        dataset = dataset.shuffle(
            min(generator.size(), buffer_size), reshuffle_each_iteration=repeat)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(
            generator.preprocess, [image, mask], [K.floatx(), K.floatx()])
    )
    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(
            generator.augment, [image, mask], [K.floatx(), K.floatx()])
    )
    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(
            generator.postprocess, [image, mask], [K.floatx(), K.floatx()])
    )
    dataset = dataset.map(
        lambda image, mask: make_shape(image, mask)
    )

    return dataset


def generate_for_augments(clazz, fold, augments, job_config, mode, path, shuffle=False, repeat=False):
    generator = DataGenerator(clazz, fold, shuffle=shuffle, path=path,
                              augmentations=augments, job_config=job_config, mode=mode)
    augmented = generator_to_dataset(generator, repeat, shuffle)
    num_images = generator.size()

    return generator, augmented, num_images
