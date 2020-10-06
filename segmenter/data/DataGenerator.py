import os
import json
import random
import numpy as np
import glob
from tensorflow.keras import backend as K
from typing import Dict
from .CachingFileReader import CachingFileReader


class DataGenerator:
    def __init__(self,
                 image_files=[],
                 preprocess=[],
                 postprocess=[],
                 augmentation_generator=None,
                 use_cache=True):

        self.cache = CachingFileReader(use_cache=use_cache)
        self.augmentation_generator = augmentation_generator
        self.image_files = image_files
        self.preprocess_functions = preprocess
        self.postprocess_functions = postprocess

    def size(self):
        return len(self.image_files)

    def preprocess(self, img, mask):
        for f in self.preprocess_functions:
            img = f(img, mask)
        return img.astype(K.floatx()), mask.astype(K.floatx())

    def postprocess(self, img, mask):
        for f in self.postprocess_functions:
            img = f(img, mask)
        return img.astype(K.floatx()), mask.astype(K.floatx())

    def augment(self, img, mask):
        augmentation_for_image = self.augmentation_generator(img)
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

    def get_image(self, idx):
        return self.image_files[idx]

    def _generate(self, idx):
        raise NotImplementedError()

    def generate(self):
        for i in range(len(self.image_files)):
            yield self._generate(i)
