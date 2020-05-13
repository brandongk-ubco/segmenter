import os
import json
import random
import numpy as np
import io
import glob
from tensorflow.keras import backend as K
from typing import Dict


class CachedFilereader:

    cache: Dict[str, bytes] = {}

    def read(self, filename):
        if filename not in self.cache:
            with open(filename, "rb") as newfile:
                self.cache[filename] = newfile.read()

        return io.BytesIO(self.cache[filename])


class DataGenerator:
    def __init__(self,
                 clazz,
                 fold=None,
                 shuffle=False,
                 mode="train",
                 path="/data",
                 augmentations=None,
                 job_config=None):

        self.cache = CachedFilereader()
        self.augmentations = augmentations
        self.path = path

        with open(os.path.join(path, "classes.json"), "r") as json_file:
            self.data = json.load(json_file)

        if mode == "evaluate" or (fold is None and mode == "val"):
            self.image_files = sorted(
                self.data["classes"][clazz]["eval_instances"])
        elif fold is None and mode == "train":
            self.image_files = sorted(
                self.data["classes"][clazz]["train_instances"])
        elif fold is not None and mode in ["train", "val"]:
            with open(
                    os.path.join(path,
                                 "%s-folds.json" % (job_config["FOLDS"])),
                    "r") as json_file:
                json_data = json.load(json_file)
                self.image_files = sorted(
                    json_data["folds"][fold][clazz][mode])
        else:
            raise ValueError("Invalid combination: mode %s / fold %s" %
                             (mode, fold))

        self.image_files = list(
            map(lambda f: os.path.abspath(os.path.join(self.path, f)),
                self.image_files))

        if shuffle:
            random.shuffle(self.image_files)

        self.job_config = job_config
        self.mask_index = [str(c)
                           for c in self.job_config["CLASSES"]].index(clazz)

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
            self.job_config["AUGMENTS"], self.data["properties"]["mean"],
            self.data["properties"]["std"], img.shape)
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
        keep_idx = np.ix_(keep.any(1)[:, 0], keep.any(0)[:, 0])
        return img[keep_idx], mask[keep_idx]

    def zero(self, img, mask):
        keep = np.logical_or(img > 0.1, mask > 0)
        img[~keep] = 0
        return img

    def _generate(self, idx):
        filename = os.path.join(self.path,
                                "{}.npz".format(self.image_files[idx]))
        i_data = np.load(self.cache.read(filename))
        img = i_data["image"]
        mask = i_data["mask"][:, :, self.mask_index]
        if len(img.shape) == 2:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        return img, mask

    def generate(self):
        for i in range(len(self.image_files)):
            yield self._generate(i)
