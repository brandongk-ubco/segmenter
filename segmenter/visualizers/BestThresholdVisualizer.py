import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from segmenter.aggregators import Aggregators


class BestThresholdVisualizer(BaseVisualizer):

    aggregator_map = dict(
        [("dummy", "")] +
        [(a.name(), a.display_name())
         for a in [Aggregators.get(a)(None) for a in Aggregators.choices()]])

    @staticmethod
    def label_bars(plot):
        for rect in plot.patches:
            height = rect.get_height()
            plot.annotate(
                '{:1.2f}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -12),  # 3 points vertical offset
                textcoords="offset points",
                ha='center',
                va='bottom')

    def execute_metrics(self):
        results = self.results.groupby("display_aggregator").agg({
            "f1-score":
            'max',
            "iou_score":
            'max'
        }).reset_index().sort_values(by=["display_aggregator"])

        plot = results.plot.bar(x="display_aggregator",
                                y=["f1-score", "iou_score"])
        BestThresholdVisualizer.label_bars(plot)

        title = "Best Metrics"
        subtitle = "{} - Class {}".format(self.label, self.clazz)

        fig = plot.get_figure()
        plt.title('')
        fig.suptitle(title, y=1.05, fontsize=14)
        plt.figtext(.5, .96, subtitle, fontsize=12, ha='center')
        plot.set_ylabel('Metric')
        plot.set_xlabel('')

        plt.legend(["F1-Score", "IOU"],
                   bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   ncol=2,
                   frameon=False)

        outfile = os.path.join(self.data_dir, "best_metrics.png")
        fig.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def execute_threshold(self):

        results = self.results.groupby("display_aggregator").agg({
            "f1-score":
            'max'
        }).reset_index()
        results = self.results.merge(results).sort_values(
            by=["display_aggregator"])

        plot = results.plot.bar(x="display_aggregator", y="threshold")
        BestThresholdVisualizer.label_bars(plot)

        title = "Best Threshold"
        subtitle = "{} - Class {}".format(self.label, self.clazz)

        fig = plot.get_figure()
        plt.title('')
        fig.suptitle(title, y=1.05, fontsize=14)
        plt.figtext(.5, .96, subtitle, fontsize=12, ha='center')
        plot.set_ylabel('Threshold')
        plot.set_xlabel('')

        plt.legend(["threshold"],
                   bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   ncol=1,
                   frameon=False)

        outfile = os.path.join(self.data_dir, "best_threshold.png")
        fig.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def execute(self):
        csv_file = os.path.join(self.data_dir, "metrics.csv")
        self.clazz = self.data_dir.split("/")[-2]
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        self.results = pd.read_csv(csv_file)
        self.results["display_aggregator"] = self.results["aggregator"].apply(
            lambda x: self.aggregator_map[x])
        self.execute_metrics()
        self.execute_threshold()
