import hashlib
import os
import psutil
import tensorflow as tf
from DataGenerator import DataGenerator
import numpy as np
from tensorflow.keras import backend as K

def hash(in_string):
    return hashlib.md5(str(in_string).encode()).hexdigest()

def get_parallel_calls():
    4 * max(psutil.cpu_count(logical=False) - 1, 1)

def make_shape(image, mask):
    image.set_shape([None, None, None])
    mask.set_shape([None, None, None])
    return image, mask

parallel_data_calls = get_parallel_calls()

def generator_to_dataset(generator, repeat, shuffle, buffer_size=500):
    dataset = tf.data.Dataset.from_generator(generator.generate, (tf.float32,tf.float32))

    if shuffle:
        dataset = dataset.shuffle(min(generator.size(), buffer_size), reshuffle_each_iteration=repeat)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(generator.augment, [image, mask], [tf.float32, tf.float32]),
        num_parallel_calls=parallel_data_calls
    )
    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(generator.postprocess, [image, mask], [K.floatx(), K.floatx()]),
        num_parallel_calls=parallel_data_calls
    )
    dataset = dataset.map(
        lambda image, mask: make_shape(image, mask),
        num_parallel_calls=parallel_data_calls
    )

    return dataset

def generate_for_augments(clazz, fold, augments, job_config, mode, method="include", path="/data", shuffle=False, repeat=False):
    generator = DataGenerator(clazz, fold, method=method, path=path, augmentations=augments, job_config=job_config, mode=mode)
    augmented = generator_to_dataset(generator, repeat, shuffle)
    num_images = generator.size()

    return generator, augmented, num_images

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        if layer_type == 'InputLayer':
            continue
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes