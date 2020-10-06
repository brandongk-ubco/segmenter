import os
import json
import random
import numpy as np
import glob
from typing import Dict
from .CachingFileReader import CachingFileReader
from .DataGenerator import DataGenerator
import random


class KFoldDataGenerator(DataGenerator):
    def __init__(self,
                 clazz,
                 fold=None,
                 shuffle=False,
                 mode="train",
                 path="/data",
                 augmentations=None,
                 job_config=None,
                 cache=True):

        self.path = path
        self.augmentations = augmentations
        self.augments = job_config["AUGMENTS"]

        with open(os.path.join(self.path, "classes.json"), "r") as json_file:
            self.data = json.load(json_file)
            self.data_properties = self.data["properties"]

        if mode == "evaluate" or (fold is None and mode == "val"):
            image_files = sorted(self.data["classes"][clazz]["eval_instances"])
        elif fold is None and mode == "train":
            image_files = sorted(
                self.data["classes"][clazz]["train_instances"])
        elif fold is not None and mode in ["train", "val"]:
            with open(
                    os.path.join(self.path,
                                 "%s-folds.json" % (job_config["FOLDS"])),
                    "r") as json_file:
                json_data = json.load(json_file)
                image_files = sorted(json_data["folds"][fold][clazz][mode])
        else:
            raise ValueError("Invalid combination: mode %s / fold %s" %
                             (mode, fold))

        image_files = list(
            map(lambda f: os.path.abspath(os.path.join(self.path, f)),
                image_files))

        if shuffle:
            random.seed(job_config["SEED"])
            random.shuffle(image_files)

        if job_config["MAX_TRAIN_SIZE"] > 0 and job_config[
                "MAX_TRAIN_SIZE"] < len(image_files):
            print("Capping %s image set at %s" %
                  (len(image_files), job_config["MAX_TRAIN_SIZE"]))
            image_files = image_files[:job_config["MAX_TRAIN_SIZE"]]

        self.mask_index = [str(c) for c in job_config["CLASSES"]].index(clazz)

        preprocess = []
        if job_config["PREPROCESS"]["ZERO_BLANKS"]:
            preprocess.append(self.zero)

        postprocess = []
        if job_config["POSTPROCESS"]["RECENTER"]:
            postprocess.append(lambda img, mask: img - np.mean(img))

        super().__init__(image_files=image_files,
                         augmentation_generator=self._augment,
                         use_cache=cache,
                         preprocess=preprocess,
                         postprocess=postprocess)

    def _augment(self, img):
        return self.augmentations(self.augments, self.data_properties["mean"],
                                  self.data_properties["std"], img.shape)

    def _generate(self, idx):
        filename = os.path.join(self.path,
                                "{}.npz".format(self.get_image(idx)))
        i_data = np.load(self.cache.read(filename))
        img = i_data["image"]
        mask = i_data["mask"][:, :, self.mask_index]
        if len(img.shape) == 2:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        return img, mask
