from segmenter.datasets.BaseDataset import BaseDataset
import nibabel as nib
import numpy as np
import random
import pandas as pd
import os
from subprocess import check_call
import numpy as np
from typing import Dict, Set, List, Any
from segmenter.helpers.contrast_stretch import contrast_stretch
import json
from skimage.transform import resize
from matplotlib import pyplot as plt


class KitsDataset(BaseDataset):

    instances: Set[str] = set()

    class_members: Dict[str, Dict[str, List]] = {
        "kidney": {
            "eval_instances": [],
            "train_instances": []
        },
        "tumor": {
            "eval_instances": [],
            "train_instances": []
        }
    }

    case: Dict[str, Any] = {
        "name": None,
        "imaging": None,
        "kidney_mask": None,
        "tumor_mask": None
    }

    def _initialize_members(self, members, membership_type):
        for instance in members:
            imaging, kidney_mask, tumor_mask = self.load_case(instance)
            for slice_idx in range(imaging.shape[0]):
                name = "{}_{}".format(instance, slice_idx)
                print(name)
                kidney_mask_slice = kidney_mask[slice_idx, :, :]
                tumor_mask_slice = tumor_mask[slice_idx, :, :]
                if self.class_represented(kidney_mask_slice):
                    self.class_members["kidney"][membership_type].append(name)
                    self.instances.add(name)
                if self.class_represented(tumor_mask_slice):
                    self.class_members["tumor"][membership_type].append(name)
                    self.instances.add(name)

    def __init__(self, src_dir):
        self.src_dir = os.path.join(os.path.abspath(src_dir), "kits19", "data")

    def initialize(self):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        check_call("git submodule update --init --remote --force --recursive",
                   shell=True,
                   cwd=current_dir)
        check_call("pip install -r requirements.txt",
                   shell=True,
                   cwd=os.path.join(current_dir, "kits19"))
        check_call("python -m starter_code.get_imaging",
                   shell=True,
                   cwd=os.path.join(current_dir, "kits19"))

    def divide_members(self):
        self.train_cases = sorted([
            d for d in os.listdir(self.src_dir) if os.path.isfile(
                os.path.join(self.src_dir, d, "segmentation.nii.gz"))
        ])

        members_cache_file = os.path.join(self.src_dir, "members.json")
        if os.path.exists(members_cache_file):
            with open(members_cache_file, "r") as infile:
                cache = json.load(infile)
                self.class_members = cache["class_members"]
                self.instances = set(cache["instances"])
            return

        members = self.train_cases.copy()
        random.shuffle(members)
        eval_instances = sorted(members[:int(len(members) / 10) + 1])
        train_instances = sorted(
            [m for m in members if m not in eval_instances])
        self._initialize_members(eval_instances, "eval_instances")
        self._initialize_members(train_instances, "train_instances")

        with open(members_cache_file, "w") as outfile:
            json.dump(
                {
                    "class_members": self.class_members,
                    "instances": list(self.instances)
                }, outfile)

    def class_represented(self, segmentation):
        return np.sum(segmentation) > 0

    def split_folds(self, class_members, num_folds):
        chunked = {}
        for clazz in self.get_classes():
            train_instances = class_members[clazz]["train_instances"]
            cases = list(set([l[:10] for l in train_instances]))
            chunked[clazz] = self.chunkify(cases, num_folds)

        folds = []
        for fold_number in range(num_folds):
            fold = {}

            for clazz, members in chunked.items():
                val_cases = [
                    c for c in class_members[clazz]["train_instances"]
                    if c[:10] in chunked[clazz][fold_number]
                ]
                train_cases = [
                    c for c in class_members[clazz]["train_instances"]
                    if c not in val_cases
                ]
                fold[clazz] = {"val": val_cases, "train": train_cases}
            folds.append(fold)
        return folds

    def load_case(self, case):
        if case != self.case["name"]:
            segmentation = nib.load(
                os.path.join(self.src_dir, case,
                             "segmentation.nii.gz")).get_fdata()
            imaging = nib.load(
                os.path.join(self.src_dir, case,
                             "imaging.nii.gz")).get_fdata()
            imaging = (imaging - imaging.min()) / np.ptp(imaging)
            kidney_mask = (segmentation == 1.).astype("float32")
            tumor_mask = (segmentation == 2.).astype("float32")
            self.case = {
                "name": case,
                "imaging": imaging.astype("float32"),
                "kidney_mask": kidney_mask.astype("float32"),
                "tumor_mask": tumor_mask.astype("float32")
            }

        return self.case["imaging"], self.case["kidney_mask"], self.case[
            "tumor_mask"]

    def get_class_members(self):
        return self.class_members

    def get_classes(self):
        return ["kidney", "tumor"]

    def iter_instances(self):
        for instance in sorted(self.instances):
            yield instance

    def get_name(self, instance):
        return instance

    def get_image(self, instance):
        case_name = "_".join(instance.split("_")[:2])
        case_idx = int(instance.split("_")[-1])
        imaging, _kidney_mask, _tumor_mask = self.load_case(case_name)
        imaging = imaging[case_idx]
        if imaging.shape[0] != 512 or imaging.shape[0] != 512:
            print("Resizing {} from {}".format(instance, imaging.shape))
            imaging = resize(imaging, output_shape=(512, 512))
        return imaging

    def get_masks(self, instance):
        case_name = "_".join(instance.split("_")[:2])
        case_idx = int(instance.split("_")[-1])
        _imaging, kidney_mask, tumor_mask = self.load_case(case_name)
        kidney_mask = kidney_mask[case_idx]
        tumor_mask = tumor_mask[case_idx]
        if kidney_mask.shape[0] != 512 or kidney_mask.shape[0] != 512:
            kidney_mask = resize(kidney_mask, output_shape=(512, 512))
            kidney_mask = (kidney_mask > 0.5).astype("float32")
        if tumor_mask.shape[0] != 512 or tumor_mask.shape[0] != 512:
            tumor_mask = resize(tumor_mask, output_shape=(512, 512))
            tumor_mask = (tumor_mask > 0.5).astype("float32")
        return np.dstack((kidney_mask, tumor_mask))

    def enhance_image(self, image):
        return contrast_stretch(image)
