from segmenter.datasets.BaseDataset import BaseDataset
from segmenter.helpers import contrast_stretch
import nibabel as nib
import numpy as np
import random
import pandas as pd
from skimage.io import imread
import os


class KitsDataset(BaseDataset):
    def __init__(self):
        raise NotImplementedError

    def get_classes(self):
        return ["kidney", "tumor"]

    def get_class_members(self):
        raise NotImplementedError

    def get_instances(self):
        raise NotImplementedError

    def get_name(self, instance):
        raise NotImplementedError

    def get_image(self, instance):
        raise NotImplementedError

    def get_mask(self, instance):
        raise NotImplementedError

    def enhance_image(self, image):
        raise NotImplementedError
