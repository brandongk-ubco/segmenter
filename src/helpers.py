import hashlib
import os
import psutil
import tensorflow as tf
from DataGenerator import DataGenerator

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

def get_parallel_calls():
    4 * max(psutil.cpu_count(logical=False) - 1, 1)

def make_shape(image, mask):
    image.set_shape([None, None, None])
    mask.set_shape([None, None, None])
    return image, mask

parallel_data_calls = get_parallel_calls()

def generator_to_dataset(generator, repeat, shuffle):
    dataset = tf.data.Dataset.from_generator(generator.generate, (tf.float32,tf.float32))

    if shuffle:
        dataset = dataset.shuffle(generator.size(), reshuffle_each_iteration=repeat)
    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(generator.augment, [image, mask], [tf.float32, tf.float32]),
        num_parallel_calls=parallel_data_calls
    )
    dataset = dataset.map(
        lambda image, mask: tf.numpy_function(generator.postprocess, [image, mask], [tf.float16, tf.float16]),
        num_parallel_calls=parallel_data_calls
    )
    dataset = dataset.map(
        lambda image, mask: make_shape(image, mask),
        num_parallel_calls=parallel_data_calls
    )
    return dataset

def generate_for_augments(clazz, fold, augments, job_config, mode, path="/data", shuffle=False, repeat=False):
    generator = DataGenerator(clazz, fold, path=path, augmentations=augments, job_config=job_config, mode=mode)
    dataset = generator_to_dataset(generator, repeat, shuffle)
    num_images = generator.size()

    return generator, dataset, num_images
