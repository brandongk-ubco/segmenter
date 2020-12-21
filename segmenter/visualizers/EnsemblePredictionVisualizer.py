import os
import json
from numpy.lib.financial import rate
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from statistics import harmonic_mean
from math import sqrt
from segmenter.config import config_from_dir
import matplotlib.cm as cm
import matplotlib


class EnsemblePredictionVisualizer(BaseVisualizer):

    configs = {}

    examples = {1: "3fdc95c42", 2: "17ac2163c", 3: "0ebb34d89", 4: "5c8104408"}

    def read_config(self, job_hash):
        if job_hash not in self.configs:
            config, hash = config_from_dir(
                os.path.join(self.data_dir, job_hash))
            assert (job_hash == hash)
            self.configs[job_hash] = config
        return self.configs[job_hash]

    def execute(self):
        ensembles_df = pd.read_csv(os.path.join(self.data_dir,
                                                "ensembles.csv"))

        for clazz in ensembles_df["CLASS"].unique():
            clazz_df = ensembles_df[ensembles_df["CLASS"] == clazz].copy()
            example = self.examples[clazz]
            predictions = None
            image = None
            mask = None
            for job_hash in clazz_df["JOB_HASH"]:
                prediction_file = os.path.join(self.data_dir, job_hash,
                                               str(int(clazz)), "results",
                                               "dummy", "0.50",
                                               "%s.npz" % example)
                prediction = np.load(prediction_file)
                if predictions is None:
                    predictions = prediction["raw_prediction"]
                else:
                    predictions = np.dstack(
                        [predictions, prediction["raw_prediction"]])
                if image is None:
                    image = prediction["image"]
                if mask is None:
                    mask = prediction["mask"]
            std = np.std(predictions, axis=2)
            std = std - np.min(std)
            std = std / np.max(std)
            mean = np.mean(predictions, axis=2)
            mean = mean - np.min(mean)
            mean = mean / np.max(mean)

            image = image - np.min(image)
            image = image / np.max(image)

            std_colors = np.dstack([std * 1, std * 0, std * 0])

            fig, ax = plt.subplots()
            plt.title(
                "Areas of Diversity in the Ensemble for {} Class {}".format(
                    example, str(int(clazz))))
            ax.imshow(image, cmap='gray')
            ax.imshow(std_colors, alpha=0.5)
            fig.set_size_inches(11, 11 * std.shape[0] / std.shape[1])
            ax.axis('off')

            outfile = os.path.join(self.data_dir,
                                   "%s_std.png" % (str(int(clazz))))
            fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)

            mean_colors = np.dstack([mean * 0, mean * 1, mean * 0])
            fig, ax = plt.subplots()
            plt.title("Ensemble Predictions for {} Class {}".format(
                example, str(int(clazz))))
            ax.imshow(image, cmap='gray')
            ax.imshow(mean_colors, alpha=0.5)
            fig.set_size_inches(11, 11 * mean.shape[0] / mean.shape[1])
            ax.axis('off')

            outfile = os.path.join(self.data_dir,
                                   "%s_mean.png" % (str(int(clazz))))
            fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
