import os
from segmenter.collectors.BaseCollector import BaseCollector
from shutil import copyfile
from segmenter.models.BestFoldWeightFinder import BestFoldWeightFinder


class WeightCollector(BaseCollector):
    def execute(self):

        classes = [
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d))
        ]

        for clazz in classes:
            fold_directory = os.path.join(os.path.abspath(self.data_dir),
                                          clazz)
            dest_dir = os.path.join(os.path.abspath(self.data_dir), clazz,
                                    "results", "weights")

            os.makedirs(dest_dir, exist_ok=True)

            weight_finder = BestFoldWeightFinder(fold_directory)

            for fold_name in sorted(weight_finder.keys()):
                weight_file = weight_finder.get(fold_name)
                extension = weight_file.split(".")[-1]
                dest_file = os.path.join(dest_dir,
                                         "{}.{}".format(fold_name, extension))
                if os.path.exists(dest_file):
                    continue
                print("{} -> {}".format(weight_file, dest_file))
                copyfile(weight_file, dest_file)
