import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
import numpy as np
from typing import Dict


class InstanceMetricsVisualizer(BaseVisualizer):

    metrics = [("dice", "F1-Score"), ("iou", "IOU")]

    aggregator_pretty = {
        "average": "Average",
        "noisy_or": "Noisy Or",
        "vote": "Vote"
    }

    def execute(self):
        csv_file = os.path.join(self.data_dir, "instance-metrics.csv")
        clazz = self.data_dir.split("/")[-2]
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        self.results = pd.read_csv(csv_file)
        for aggregator in self.results["aggregator"].unique():
            aggregator_results = self.results[self.results.aggregator ==
                                              aggregator]
            for threshold in aggregator_results["threshold"].unique():
                threshold_results = aggregator_results[
                    aggregator_results.threshold == threshold]
                subtitle = "Class {}, {} aggregation with threshold {}".format(
                    clazz, aggregator, threshold)
                for metric, display in self.metrics:
                    outfile = os.path.join(
                        os.path.dirname(self.data_dir), "results", aggregator,
                        "{:.2f}".format(threshold),
                        "instance-metrics-{}.png".format(metric))
                    print(outfile)
                    metric_results = threshold_results[metric]
                    metric_plot = self.visualize(metric_results, display)
                    title = r'{} ($\mu$ = {:.2f}, $\sigma$ = {:.2f})'.format(
                        display, np.mean(metric_results),
                        np.std(metric_results))
                    metric_plot.suptitle(title, y=1.07, fontsize=16)
                    plt.figtext(.5, .98, subtitle, fontsize=14, ha='center')
                    plt.savefig(outfile,
                                dpi=100,
                                bbox_inches='tight',
                                pad_inches=0.5)
                    plt.close()
            for metric, display in self.metrics:
                plot = self.results.boxplot(column=[metric],
                                            by='threshold',
                                            grid=False)
                title = "{} by Threshold".format(display)
                subtitle = "Class {}, {} aggregation".format(clazz, aggregator)

                fig = plot.get_figure()
                plt.title('')
                fig.suptitle(title, y=1.05, fontsize=14)
                plt.figtext(.5, .96, subtitle, fontsize=12, ha='center')
                plot.set_ylabel(display)
                plot.set_xlabel('Threshold')
                plot.tick_params(axis='x', rotation=90)

                outfile = os.path.join(
                    os.path.dirname(self.data_dir), "results",
                    "{}-instance-metrics-{}.png".format(aggregator, metric))
                fig.savefig(outfile,
                            dpi=100,
                            bbox_inches='tight',
                            pad_inches=0.5)
                plt.close()

    def visualize(self, ratings, display_name):
        fig, (ax1, ax2) = plt.subplots(1,
                                       2,
                                       gridspec_kw={'width_ratios': [3, 1]})
        fig.tight_layout()
        bins = np.linspace(0, 1, num=51)

        fig.set_size_inches(10, 5)

        ax1.hist(ratings,
                 bins=bins,
                 weights=100 * np.ones(len(ratings)) / len(ratings))
        ax1.set(xlabel=display_name, ylabel="Frequency (%)")

        plt.subplots_adjust(hspace=.2)

        ax2.boxplot(ratings, vert=True)
        ax2.set(ylabel=display_name)
        ax2.set_xticks([])

        return fig
