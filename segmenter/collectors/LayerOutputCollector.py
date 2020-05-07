import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.collectors.BaseCollector import BaseCollector
import glob
import numpy as np
from typing import Dict


class LayerOutputCollector(BaseCollector):

    layer_types = ["activation", "convolutional", "normalization"]
    results = pd.DataFrame()

    def execute(self):
        results = self.collect_results(self.data_dir)
        for result in sorted(results):
            print(result)
            fold_results = np.load(result)
            fold = result.split("/")[-2]
            for layer_type in self.layer_types:
                layer_results = pd.DataFrame([fold_results[layer_type]])
                layer_results["layer_type"] = layer_type
                layer_results["fold"] = fold
                self.results = self.results.append(layer_results)

        self.results.to_csv(os.path.join(self.data_dir, "layer-outputs.csv"),
                            index=False)

    def collect_results(self, directory):
        return glob.glob("{}/**/layer_outputs.npz".format(directory),
                         recursive=True)
