import os
import json
import pandas as pd
import numpy as np
from segmenter.collectors.BaseCollector import BaseCollector
import glob


class MetricCollector(BaseCollector):

    results = pd.DataFrame()

    def execute(self):
        for result in self.collect_results(self.data_dir):
            print(result)
            aggregator = result.split("/")[-3]
            threshold = float(result.split("/")[-2])
            with open(result, "r") as sample_file:
                sample_results = json.load(sample_file)
            sample_results["threshold"] = threshold
            sample_results["aggregator"] = aggregator
            df = pd.DataFrame.from_dict([sample_results])
            self.results = self.results.append(df, ignore_index=True)
        if not self.results.empty:
            print("Writing {}".format(
                os.path.join(self.data_dir, "results.csv")))
            self.results.to_csv(os.path.join(self.data_dir, "results.csv"),
                                index=False)

    def collect_results(self, directory):
        return glob.glob("{}/**/results.json".format(directory),
                         recursive=True)