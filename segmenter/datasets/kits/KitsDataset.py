from segmenter.datasets.BaseDataset import BaseDataset
from segmenter.helpers import contrast_stretch
import nibabel as nib
import numpy as np
import random
import pandas as pd
from skimage.io import imread
import os
from subprocess import check_call


class KitsDataset(BaseDataset):
    def initialize(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        check_call("git submodule update --init", shell=True, cwd=current_dir)
        check_call("pip install -r requirements.txt",
                   shell=True,
                   cwd=os.path.join(current_dir, "kits19"))
        check_call("python -m starter_code.get_imaging",
                   shell=True,
                   cwd=os.path.join(current_dir, "kits19"))

    def get_classes(self):
        return ["kidney", "tumor"]

    def get_class_members(self):
        raise NotImplementedError

    def iter_instances(self):
        raise NotImplementedError

    def get_name(self, instance):
        raise NotImplementedError

    def get_image(self, instance):
        raise NotImplementedError

    def get_masks(self, instance):
        raise NotImplementedError

    def enhance_image(self, image):
        raise NotImplementedError
