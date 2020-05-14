import os
import json
import pandas as pd
import numpy as np
from segmenter.collectors.BaseCollector import BaseCollector
from shutil import copyfile
from segmenter.models.BestFoldWeightFinder import BestFoldWeightFinder


class WeightCollector(BaseCollector):
    def execute(self):
        os.makedirs(os.path.join(self.data_dir, "weights"), exist_ok=True)
        fold_directory = os.path.join(os.path.abspath(self.data_dir), "..")
        weight_finder = BestFoldWeightFinder(fold_directory)

        for fold_name in sorted(weight_finder.keys()):
            weight_file = weight_finder.get(fold_name)
            extension = weight_file.split(".")[-1]
            dest_file = os.path.join(self.data_dir, "weights",
                                     "{}.{}".format(fold_name, extension))
            if os.path.exists(dest_file):
                continue
            print("{} -> {}".format(weight_file, dest_file))
            copyfile(weight_file, dest_file)
