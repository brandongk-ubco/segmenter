import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from segmenter.aggregators import Aggregators


class AUCVisualizer(BaseVisualizer):
    def execute(self):
        csv_file = os.path.join(self.data_dir, "metrics.csv")
        clazz = self.data_dir.split("/")[-2]
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        self.results = pd.read_csv(csv_file)
        self.results["false_positive_rate"] = 1 - self.results["specificity"]
        self.results["false_discovery_rate"] = 1 - self.results["precision"]
        self.results = self.results.round(2)
        self.results = self.results[[
            "recall", "false_positive_rate", "false_discovery_rate",
            "aggregator"
        ]]
        for aggregator_name in self.results["aggregator"].unique():
            aggregator = Aggregators.get(aggregator_name)(self.job_config)
            aggregator_results = self.results[
                self.results.aggregator == aggregator_name][[
                    "recall", "false_positive_rate", "false_discovery_rate"
                ]]
            aggregator_results = aggregator_results.groupby("recall").agg(
                'min').reset_index()
            aggregator_results = aggregator_results.append(
                {
                    "recall": 0.,
                    "false_positive_rate": 0.,
                    "false_discovery_rate": 0.
                },
                ignore_index=True)
            aggregator_results = aggregator_results.append(
                {
                    "recall": 1.,
                    "false_positive_rate": 1.,
                    "false_discovery_rate": 1.
                },
                ignore_index=True)
            aggregator_results = aggregator_results.sort_values("recall")
            recall = aggregator_results["recall"]
            fpr = aggregator_results["false_positive_rate"]
            fdr = aggregator_results["false_discovery_rate"]

            auc_fpr = np.trapz(recall, fpr)
            auc_fdr = np.trapz(recall, fdr)

            #Plot False-Positive Rate
            fig, ax = self.visualize(recall, fpr)
            ax.set_xlabel('False Positive Rate (1 - Specificity)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
            subtitle = "Class {}, {} aggregation".format(
                clazz, aggregator.display_name())
            plt.figtext(.5, .95, subtitle, fontsize=14, ha='center')
            plt.title(
                'True Positive Rate vs. False Positive Rate (AUC = {:1.2f})'.
                format(round(auc_fpr, 3)),
                y=1.15,
                fontsize=16)
            outfile = os.path.join(
                self.data_dir,
                "{}-auc-false-positive.png".format(aggregator.name()))
            print(outfile)
            plt.savefig(outfile, dpi=100, bbox_inches='tight', pad_inches=0.5)
            plt.close()

            #Plot False-Discovery Rate
            fig, ax = self.visualize(recall, fdr)
            ax.set_xlabel('False Discovery Rate (1 - Precision)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
            subtitle = "Class {}, {} aggregation".format(
                clazz, aggregator.display_name())
            plt.figtext(.5, .95, subtitle, fontsize=14, ha='center')
            plt.title(
                'True Positive Rate vs. False Discovery Rate (AUC = {:1.2f})'.
                format(round(auc_fdr, 3)),
                y=1.15,
                fontsize=16)
            outfile = os.path.join(
                self.data_dir,
                "{}-auc-false-discovery.png".format(aggregator.name()))
            print(outfile)
            plt.savefig(outfile, dpi=100, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def visualize(self, tpr, fpr):
        f, ax = plt.subplots()
        ax.plot(fpr, tpr, marker='o')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, max(fpr)])
        return f, ax
