import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.collectors.BaseCollector import BaseCollector
import glob
import numpy as np
from typing import Dict
from segmenter.helpers.p_tqdm import t_map


class TrainCollector(BaseCollector):

    results = pd.DataFrame()

    def execute_result(self, result):
        clazz = result.split("/")[-6]
        fold_str = result.split("/")[-3]
        fold = fold_str.split("b")[0].replace("fold", "")
        boost_fold = int(fold_str.split("b")[1]) if "b" in fold_str else 0
        result_df = pd.read_csv(result)
        result_df["class"] = clazz
        result_df["fold"] = fold
        result_df["boost_fold"] = boost_fold
        self.results = self.results.append(result_df)

    def execute(self):
        outfile = os.path.join(self.data_dir, "train_results.csv")
        if os.path.exists(outfile):
            print("Training results already collected in {}".format(
                self.data_dir))
            return

        t_map(self.execute_result, self.collect_results(self.data_dir))

        if not self.results.empty:
            print("Writing {}".format(outfile))
            self.results.to_csv(outfile, index=False)

    def collect_results(self, directory):
        return glob.glob("{}/../**/train.csv".format(directory),
                         recursive=True)
