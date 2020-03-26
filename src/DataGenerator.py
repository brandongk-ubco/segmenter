import os
import json
import random
import numpy as np
import io
import glob
from tensorflow.keras import backend as K

class CachedFilereader:

    cache = {}

    def read(self, filename):
        if filename not in self.cache:
            with open(filename, "rb") as newfile:
                self.cache[filename] = newfile.read()

        return io.BytesIO(self.cache[filename])

cache = CachedFilereader()

class DataGenerator:

    def __init__(self, clazz, fold=None, shuffle=False, mode="train", path="/data", augmentations=None, job_config=None):

        self.augmentations = augmentations
        self.path = path

        with open(os.path.join(path, "classes.json"), "r") as json_file:
            self.data = json.load(json_file)

        if fold is None and mode == "evaluate":
            self.image_files = sorted(self.data["classes"][clazz]["eval_instances"])
        elif fold is None and mode == "train":
            self.image_files = sorted(self.data["classes"][clazz]["train_instances"])
        elif fold is None and mode == "val":
            self.image_files = sorted(self.data["classes"][clazz]["eval_instances"])
        elif fold is not None and mode in ["train", "val"]:
            with open(os.path.join(path, "%s-folds.json" % (job_config["FOLDS"])), "r") as json_file:
                json_data = json.load(json_file)
                self.image_files = sorted(json_data["folds"][fold][clazz][mode])
        else:
            raise ValueError("Invalid combination: mode %s / fold %s" % (mode, fold))
            
        self.image_files = list(map(lambda f: os.path.abspath(os.path.join(self.path, f)), self.image_files))

        if shuffle:
            random.shuffle(self.image_files)

        self.job_config = job_config
        self.mask_index = [str(c) for c in self.job_config["CLASSES"]].index(clazz)

    def size(self):
        return len(self.image_files)

    def preprocess(self, img, mask):
        if True or self.job_config["PREPROCESS"]["ZERO_BLANKS"]:
            img = self.zero(img, mask)
        return img.astype(K.floatx()), mask.astype(K.floatx())

    def postprocess(self, img, mask):
        if False and self.job_config["POSTPROCESS"]["RECENTER"]:
            img = img - np.mean(img)
        return img.astype(K.floatx()), mask.astype(K.floatx())

    def augment(self, img, mask):
        augmentation_for_image = self.augmentations(
            self.job_config["AUGMENTS"],
            self.data["properties"]["mean"],
            self.data["properties"]["std"],
            img.shape
        )
        if augmentation_for_image is None:
            return img, mask
        mask_coverage_before = np.sum(mask)
        mask_coverage_after = 0
        while mask_coverage_after / mask_coverage_before < 0.5:
            augmented = augmentation_for_image(image=img, mask=mask)
            mask_coverage_after = np.sum(augmented['mask'])

        return augmented["image"], augmented["mask"]

    # This works, but we can't stack images of different dimensions without padding...
    # So, this will only work in practice for a batch size of 1....
    def crop(self, img, mask):
        keep = img > 0.1
        keep_idx = np.ix_(keep.any(1)[:,0], keep.any(0)[:,0])
        return img[keep_idx], mask[keep_idx]

    def zero(self, img, mask):
        keep = np.logical_or(img > 0.1, mask > 0)
        img[~keep] = 0
        return img

    def _generate(self, idx):
        filename = os.path.join(self.path, "{}.npz".format(self.image_files[idx]))
        i_data = np.load(cache.read(filename))
        img = i_data["image"]
        mask = i_data["mask"][:, :, self.mask_index]
        if len(img.shape) == 2:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        return img, mask

    def generate(self):
        for i in range(len(self.image_files)):
            yield self._generate(i)

if __name__ == "__main__":
    import os
    from augmentations import train_augments, val_augments
    from matplotlib import pyplot as plt
    from helpers import generate_for_augments, hash
    import pprint
    import os
    import tensorflow as tf
    from config import get_config

    path = os.path.abspath(os.environ.get("DATA_PATH", "/data"))
    outdir = os.path.abspath(os.environ.get("OUTPUT_DIRECTORY", "/output"))

    job_config = get_config()
    job_hash = hash(job_config)

    pprint.pprint(job_config)

    K.set_floatx(job_config["PRECISION"])
    clazz = os.environ.get("CLASS", job_config["CLASSES"][0])


    output_folder = os.path.join(outdir, job_hash, clazz, "augmented")
    print("Using directory %s" % output_folder)
    os.makedirs(output_folder, exist_ok=True)

    val_generators, augmented_val_dataset, num_val_images = generate_for_augments(clazz, None, val_augments, job_config, path=path, mode="val", repeat=True)
    train_generators, augmented_train_dataset, num_train_images = generate_for_augments(clazz, None, train_augments, job_config, path=path, mode="train", shuffle=True, repeat=True)

    print("Found %s training images" % num_train_images)
    print("Found %s validation images" % num_val_images)

    seen_imgs = []
    repeated = 0
    for i, (augmented_img, augmented_mask) in enumerate(augmented_train_dataset.as_numpy_iterator()):
        if i >= 2*num_train_images:
            break

        augmented_img = augmented_img[:,:,0]
        augmented_mask = augmented_mask[:,:,0]

        been_seen = False
        for seen in seen_imgs:
            if (seen==augmented_img).all():
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
        plt.subplot(2,1,1)
        plt.imshow(augmented_img.astype('float32'), cmap='gray')
        plt.subplot(2,1,2)
        plt.imshow(augmented_mask.astype('float32'), cmap='gray')
        plt.savefig(os.path.join(output_folder, "{}.png".format(i)))
        plt.close()

    # This should be roughly 25% for 50% rotation augments
    # This should be 100% for val augments
    print("%s/%s" % (repeated, num_train_images))
