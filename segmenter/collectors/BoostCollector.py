import os
import json
import pandas as pd
import numpy as np
from segmenter.collectors.BaseCollector import BaseCollector
import glob
from segmenter.models import find_best_weight
from segmenter.helpers.parse_fold import parse_fold


class BoostCollector(BaseCollector):
    def execute(self):
        results = []
        for fold_dir in self.collect_results(self.data_dir):
            best_weight = find_best_weight(fold_dir)
            fold, boost_fold = parse_fold(fold_dir.split("/")[-1])
            loss = float(best_weight.split("-")[-1][:-3])
            clazz = fold_dir.split("/")[-4]
            results.append({
                "fold": fold,
                "boost_fold": boost_fold,
                "loss": loss,
                "class": clazz
            })
        df = pd.DataFrame.from_dict(results).sort_values(
            ["fold", "boost_fold"])
        df["baseline"] = df.apply(lambda x: df[
            (df["fold"] == x["fold"]) & (df["boost_fold"] == 0)].iloc[0].loss,
                                  axis=1)
        df["improvement"] = df.apply(
            lambda x: 100 * (x["baseline"] - x["loss"]) / x["baseline"],
            axis=1)
        if not df.empty:
            print("Writing {}".format(os.path.join(self.data_dir,
                                                   "boost.csv")))
            df.to_csv(os.path.join(self.data_dir, "boost.csv"), index=False)

    def collect_results(self, directory):
        directory = os.path.join(os.path.abspath(directory), "..")
        return [
            os.path.join(directory, d) for d in os.listdir(directory) if
            os.path.isdir(os.path.join(directory, d)) and d not in ["results"]
        ]
