import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.collectors.BaseCollector import BaseCollector
import glob
import numpy as np
from typing import Dict


class InstanceMetricsCollector(BaseCollector):

    results = pd.DataFrame()

    def execute(self):
        results = self.collect_results(self.data_dir)
        for result in sorted(results):
            print(result)
            with open(result, "r") as results_file:
                ratings = json.load(results_file)
            aggregator = result.split("/")[-3]
            threshold = result.split("/")[-2]
            pd_results = pd.DataFrame.from_dict(ratings).T
            pd_results["aggregator"] = aggregator
            pd_results["threshold"] = threshold
            self.results = self.results.append(pd_results)
        self.results.to_csv(os.path.join(self.data_dir,
                                         "instance-metrics.csv"),
                            index=False)

    def collect_results(self, directory):
        return glob.glob("{}/**/predictions.json".format(directory),
                         recursive=True)
