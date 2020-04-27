import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
import numpy as np
from typing import Dict


class ReportVisualizer(BaseVisualizer):

    ratings: Dict[str, Dict[str, Dict[str, float]]] = {}

    def execute(self):
        results = self.collect_results(self.data_dir)
        for result in results:
            print(result)
            with open(result, "r") as results_file:
                ratings = json.load(results_file)
            clazz = result.split("/")[-5]
            aggregator = result.split("/")[-3]
            threshold = result.split("/")[-2]
            subtitle = "Class {}, {} with threshold {}".format(
                clazz, aggregator, threshold)
            for key, display in [("iou", "Intersection-Over-Union"),
                                 ("dice", "F1-Score")]:
                metric_results = [v[key] for (k, v) in ratings.items()]
                metric_plot = self.visualize(metric_results, display)
                title = r'{} ($\mu$ = {:.2f}, $\sigma$ = {:.2f})'.format(
                    display, np.mean(metric_results), np.std(metric_results))
                metric_plot.suptitle(title, y=1.07, fontsize=16)
                plt.figtext(.5, .98, subtitle, fontsize=14, ha='center')
                plt.savefig(os.path.join(
                    os.path.dirname(result),
                    "predictions-{}-{}-{}-{}.png".format(
                        clazz, aggregator, threshold, key)),
                            dpi=100,
                            bbox_inches='tight',
                            pad_inches=0.5)
                plt.close()

    def collect_results(self, directory):
        return glob.glob("{}/**/predictions.json".format(directory),
                         recursive=True)

    def metrics(self, mask, prediction):
        boolean_mask = mask.astype(bool)
        boolean_prediction = prediction.astype(bool)

        intr = np.sum(np.logical_and(boolean_prediction, boolean_mask))
        union = np.sum(np.logical_or(boolean_prediction, boolean_mask))
        iou = intr / union

        dice = 2 * intr / (np.sum(boolean_prediction) + np.sum(boolean_mask))

        return iou, dice

    def visualize(self, ratings, display_name):
        fig, (ax1, ax2) = plt.subplots(1,
                                       2,
                                       gridspec_kw={'width_ratios': [3, 1]})
        fig.tight_layout()
        bins = np.linspace(0, 1, num=51)

        fig.set_size_inches(10, 5)

        ax1.hist(ratings, bins=bins)
        ax1.set(xlabel=display_name, ylabel="Count")

        plt.subplots_adjust(hspace=.2)

        ax2.boxplot(ratings, vert=True)
        ax2.set(ylabel=display_name)
        ax2.set_xticks([])

        # plt.subplots_adjust(wspace=0, hspace=0)

        return fig
