import numpy as np
import random
import pandas as pd
from skimage.io import imread
import os

from segmenter.datasets.BaseDataset import BaseDataset
from segmenter.datasets.severstal.helpers import build_mask
from segmenter.helpers import contrast_stretch


class SeverstalDataset(BaseDataset):
    def __init__(self, src_dir):
        self.path = os.path.abspath(src_dir)
        # reading in the training set
        data = pd.read_csv(os.path.join(self.path, 'train.csv'))

        # isolating the file name and class
        data['ImageId'], data['ClassId'] = data.ImageId_ClassId.str.split(
            '_', n=1).str
        data['ClassId'] = data['ClassId'].astype(np.uint8)

        # keep only the images with labels
        self.filtered = data.dropna(subset=['EncodedPixels'], axis='rows')

        # squash multiple rows per image into a list
        self.squashed = self.filtered[['ImageId', 'EncodedPixels', 'ClassId']] \
                    .groupby('ImageId', as_index=False) \
                    .agg(list)

        self.no_defects = data[[
            'ImageId'
        ]].drop_duplicates().set_index('ImageId').drop(
            self.squashed.ImageId.tolist()).index.tolist()
        print("No defect instances: %s" % len(self.no_defects))

    def get_classes(self):
        return sorted([str(c) for c in self._get_classes()])

    def _get_classes(self):
        return sorted([
            c for c in self.filtered['ClassId'].sort_index().unique().tolist()
        ])

    def get_class_members(self):
        classes = {}
        for clazz in self._get_classes():
            members = self.filtered[self.filtered["ClassId"] ==
                                    clazz]["ImageId"].tolist()
            members = [os.path.splitext(m)[0] for m in members]
            random.shuffle(members)
            eval_instances = members[:int(len(members) / 10) + 1]
            train_instances = [m for m in members if m not in eval_instances]
            classes[str(clazz)] = {
                "eval_instances": eval_instances,
                "train_instances": train_instances
            }
        return classes

    def iter_instances(self):
        for _idx, instance in self.squashed.iterrows():
            yield instance

    def get_name(self, instance):
        return instance['ImageId'][:-4]

    def get_image(self, instance):
        return imread(os.path.join(self.path, "train", instance['ImageId']),
                      as_gray=True).astype(np.float32)

    def get_masks(self, instance):
        return build_mask(instance['EncodedPixels'],
                          instance['ClassId']).astype(np.float32)

    def enhance_image(self, image):
        return contrast_stretch(image)
