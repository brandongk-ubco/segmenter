import os
import json
import pandas as pd
import numpy as np
from segmenter.collectors.BaseCollector import BaseCollector
import glob


class MetricCollector(BaseCollector):

    results = pd.DataFrame()

    def execute(self):
        outfile = os.path.join(self.data_dir, "metrics.csv")
        if os.path.exists(outfile):
            print("Metrics already collected in {}".format(self.data_dir))
            return
        for result in sorted(self.collect_results(self.data_dir)):
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
            print("Writing {}".format(outfile))
            self.results.to_csv(outfile, index=False)

    def collect_results(self, directory):
        return glob.glob("{}/**/results.json".format(directory),
                         recursive=True)
