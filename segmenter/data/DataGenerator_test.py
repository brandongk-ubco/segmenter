import os
from segmenter.augmentations import train_augments, val_augments
from matplotlib import pyplot as plt
from segmenter.data import augmented_generator
import pprint
import os
import tensorflow as tf
from segmenter.config import get_config
from tensorflow.keras import backend as K
import numpy as np
from typing import List

if __name__ == "__main__":
    path = os.path.abspath(os.environ.get("DATA_PATH", "/data"))
    outdir = os.path.abspath(os.environ.get("OUTPUT_DIRECTORY", "/output"))

    job_config, job_hash = get_config(path, outdir)
    pprint.pprint(job_config)

    K.set_floatx(job_config["PRECISION"])
    clazz = os.environ.get("CLASS", job_config["CLASSES"][0])

    output_folder = os.path.join(outdir, job_hash, clazz, "augmented")
    print("Using directory %s" % output_folder)
    os.makedirs(output_folder, exist_ok=True)

    val_generators, augmented_val_dataset, num_val_images = augmented_generator(
        clazz,
        None,
        val_augments,
        job_config,
        path=path,
        mode="val",
        repeat=True)
    train_generators, augmented_train_dataset, num_train_images = augmented_generator(
        clazz,
        None,
        train_augments,
        job_config,
        path=path,
        mode="train",
        shuffle=True,
        repeat=True)

    print("Found %s training images" % num_train_images)
    print("Found %s validation images" % num_val_images)

    seen_imgs: List[np.ndarray] = []
    repeated = 0
    for i, (augmented_img, augmented_mask) in enumerate(
            augmented_train_dataset.as_numpy_iterator()):
        if i >= 2 * num_train_images:
            break

        augmented_img = augmented_img[:, :, 0]
        augmented_mask = augmented_mask[:, :, 0]

        been_seen = False
        for seen in seen_imgs:
            if (seen == augmented_img).all():
                been_seen = True
                break

        if not been_seen:
            seen_imgs.append(augmented_img)

        if not been_seen and i > num_train_images:
            print("New image in 2nd iteration!")

        if been_seen and i < num_train_images:
            print("Repeated image in 1st iteration!")

        if been_seen and i > num_train_images:
            repeated += 1
            print("Repeated image in 2nd iteration!")

        print(i)
        plt.subplot(2, 1, 1)
        plt.imshow(augmented_img.astype('float32'), cmap='gray')
        plt.subplot(2, 1, 2)
        plt.imshow(augmented_mask.astype('float32'), cmap='gray')
        plt.savefig(os.path.join(output_folder, "{}.png".format(i)))
        plt.close()

    # This should be roughly 25% for 50% rotation augments
    # This should be 100% for val augments
    print("%s/%s" % (repeated, num_train_images))
