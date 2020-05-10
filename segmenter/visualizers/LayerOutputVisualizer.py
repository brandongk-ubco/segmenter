import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
import numpy as np


class LayerOutputVisualizer(BaseVisualizer):

    bins = np.linspace(-10, 10, num=2001)

    def execute(self):
        csv_file = os.path.join(self.data_dir, "layer-outputs.csv")
        clazz = self.data_dir.split("/")[-2]

        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        self.results = pd.read_csv(csv_file)
        for layer_type in self.results["layer_type"].unique():
            layer_type_results = self.results.copy()[self.results["layer_type"]
                                                     == layer_type]
            layer_type_results.drop("layer_type", axis=1, inplace=True)
            layer_type_results.drop("fold", axis=1, inplace=True)
            layer_type_results = layer_type_results.sum(axis=0)

            weights = 100 * layer_type_results / np.sum(layer_type_results)

            fig = plt.figure()
            bins = self.bins[:len(weights)]
            plt.hist(bins, self.bins, weights=weights)
            weighted_bins = np.multiply(weights, bins)
            percentile = np.percentile(weights, 99.9)
            plt.xlabel("Output Value")
            plt.ylabel("Frequency (%)")
            plt.ylim([0, percentile])

            title = "Output Histogram for {} layers".format(layer_type)
            subtitle1 = "{} - Class {}".format(self.label, clazz)
            subtitle2 = "Frequency Peak of {:1.2f}% at {:1.2f}.  99.9th Percentile of {:1.2f}%.".format(
                np.max(weights), self.bins[np.argmax(weights)], percentile)
            subtitle3 = "Output Value Mean {:1.2f}, Median {:1.2f}, St. Dev. {:1.2f}".format(
                np.mean(weighted_bins), np.median(weighted_bins),
                np.std(weighted_bins))
            plt.title('')
            fig.suptitle(title, y=1.06, fontsize=14)
            plt.figtext(.5, 0.99, subtitle1, fontsize=12, ha='center')
            plt.figtext(.5, .95, subtitle2, fontsize=12, ha='center')
            plt.figtext(.5, .91, subtitle3, fontsize=12, ha='center')
            outfile = os.path.join(self.data_dir,
                                   "layer-output-{}.png".format(layer_type))
            print(outfile)
            plt.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def visualize(self, result):
        plot = plt.plot([1], [1])
        return plot
