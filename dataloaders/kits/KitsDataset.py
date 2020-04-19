import cv2 as cv
import numpy as np
import random
import pandas as pd
from skimage.io import imread
import os
import sys
sys.path.append("..")

from BaseDataset import BaseDataset
from enhancements import contrast_stretch
import nibabel as nib

current_dir = os.path.dirname(os.path.realpath(__file__))


class KitsDataset(BaseDataset):
    def __init__(self, path=current_dir, class_coverage_min=0.1):
        self.path = os.path.abspath(os.path.join(path, "kits19", "data"))
        self.cases = [
            d for d in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, d)) and os.path.isfile(
                os.path.join(self.path, d, "segmentation.nii.gz"))
        ]

        for case in self.cases:
            image, segmentation = self._load_case(case)

    def _load_case(self, case):
        image = nib.load(os.path.join(self.path, case,
                                      "imaging.nii.gz")).get_data()
        segmentation = nib.load(
            os.path.join(self.path, case, "segmentation.nii.gz")).get_data()
        return image, segmentation

    def get_classes(self):
        return ["kidney", "tumor"]

    def get_class_members(self):
        return classes

    def get_instances(self):
        for instance in self.cases:
            yield instance

    def get_name(self, instance):
        return instance['ImageId'][:-4]

    def get_image(self, instance):
        return imread(os.path.join(self.path, "train", instance['ImageId']),
                      as_gray=True).astype(np.float32)

    def get_mask(self, instance):
        return build_mask(instance['EncodedPixels'],
                          instance['ClassId']).astype(np.float32)

    def enhance_image(self, image):
        return contrast_stretch(image)