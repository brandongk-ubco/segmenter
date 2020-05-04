import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from segmenter.aggregators import Aggregators


class MetricVisualizer(BaseVisualizer):

    series = [("f1-score", "F1-Score"), ("iou_score", "IOU"),
              ("precision", "Precision"), ("recall", "Recall")]

    def execute(self):
        csv_file = os.path.join(self.data_dir, "metrics.csv")
        clazz = self.data_dir.split("/")[-2]
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        self.results = pd.read_csv(csv_file)
        for aggregator_name in self.results["aggregator"].unique():
            aggregator = Aggregators.get(aggregator_name)(self.job_config)
            aggregator_results = self.results[self.results.aggregator ==
                                              aggregator_name].sort_values(
                                                  "threshold")
            f1_max = max(aggregator_results["f1-score"])
            title = "Metrics vs. Threshold"
            subtitle = "Class {}, {} Aggregator (Max F1-Score {:1.2f})".format(
                clazz, aggregator.display_name(), f1_max)

            outfile = os.path.join(self.data_dir,
                                   "{}-metrics.png".format(aggregator_name))
            print(outfile)
            plot = self.visualize(aggregator_results)
            plot.suptitle(title, y=1.05, fontsize=16)
            plt.figtext(.5, .96, subtitle, fontsize=14, ha='center')
            plot.savefig(outfile, dpi=100, bbox_inches='tight', pad_inches=0.5)
            plot.close()

    def visualize(self, results):
        for series, _display in self.series:
            plt.plot(results["threshold"], results[series], marker='o')
        plt.legend([d for (s, d) in self.series],
                   bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   ncol=len(self.series),
                   frameon=False)
        plt.xlabel("Threshold")
        plt.ylabel("Metrics")
        return plt
