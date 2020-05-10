import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.collectors.BaseCollector import BaseCollector
import glob
import numpy as np
from typing import Dict
from segmenter.helpers.p_tqdm import p_map


class PredictionCollector(BaseCollector):

    ratings: Dict[str, Dict[str, Dict[str, float]]] = {}

    def execute_result(self, result):
        name = os.path.basename(result)[:-4]
        if name == "layer_outputs":
            return

        directory = os.path.dirname(result)
        r = np.load(result)
        iou, dice = self.metrics(r["mask"], r["prediction"])
        if directory not in self.ratings:
            self.ratings[directory] = {}
        self.ratings[directory][name] = {"iou": iou, "dice": dice}

    def execute(self):
        results = sorted(self.collect_results(self.data_dir))
        p_map(self.execute_result, results)

        # Persist the report to disk
        for directory, ratings in self.ratings.items():
            with open(os.path.join(directory, "predictions.json"),
                      "w") as results_json:
                json.dump(ratings, results_json)

    def collect_results(self, directory):
        return glob.glob("{}/**/*.npz".format(directory), recursive=True)

    def metrics(self, mask, prediction):
        boolean_mask = mask.astype(bool)
        boolean_prediction = prediction.astype(bool)

        intr = np.sum(np.logical_and(boolean_prediction, boolean_mask))
        union = np.sum(np.logical_or(boolean_prediction, boolean_mask))
        iou = intr / union

        dice = 2 * intr / (np.sum(boolean_prediction) + np.sum(boolean_mask))

        return iou, dice
