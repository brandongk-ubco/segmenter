import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer


class MetricVisualizer(BaseVisualizer):

    series = [("f1-score", "F1-Score"), ("iou_score", "IOU"),
              ("precision", "Precision"), ("recall", "Recall")]

    aggregator_pretty = {
        "average": "Average",
        "noisy_or": "Noisy Or",
        "vote": "Vote"
    }

    def execute(self):
        self.collect_results()

    def collect_results(self):
        csv_file = os.path.join(self.data_dir, "results.csv")
        clazz = self.data_dir.split("/")[-2]
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        self.results = pd.read_csv(csv_file)
        for aggregator in self.results["aggregator"].unique():
            aggregator_results = self.results[
                self.results.aggregator == aggregator].sort_values("threshold")
            title = "Metrics vs. Threshold"
            subtitle = "Class {}, {} Aggregator".format(
                clazz, self.aggregator_pretty.get(aggregator, aggregator))

            outfile = os.path.join(self.data_dir, "{}.png".format(aggregator))
            print(outfile)
            plot = self.visualize(aggregator_results)
            plot.suptitle(title, y=1.05, fontsize=16)
            plt.figtext(.5, .96, subtitle, fontsize=14, ha='center')
            plot.savefig(outfile, dpi=100, bbox_inches='tight', pad_inches=0.5)
            plot.close()

    def visualize(self, results):
        thresholds = results["threshold"]
        for series, _display in self.series:
            series_results = results[series]
            plt.plot(thresholds, series_results, marker='o')
        plt.legend([d for (s, d) in self.series],
                   bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   ncol=len(self.series),
                   frameon=False)
        return plt
