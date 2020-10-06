import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from segmenter.aggregators import Aggregators
from segmenter.helpers.p_tqdm import t_map as mapper
import glob


class CombinedAUCVisualizer(BaseVisualizer):

    aggregator_map = dict(
        [("dummy", "Dummy")] +
        [(a.name(), a.display_name())
         for a in [Aggregators.get(a)(None) for a in Aggregators.choices()]])

    dataset_combined_visualizer = True

    def execute_result(self, result):
        result_data = pd.read_csv(result)

        job_hash = result.split("/")[-4]
        dataset = result.split("/")[-5]
        clazz = result.split("/")[-3]

        result_data["false_positive_rate"] = 1 - result_data["specificity"]
        result_data["false_discovery_rate"] = 1 - result_data["precision"]
        result_data = result_data.round(2)
        result_data = result_data[[
            "recall", "false_positive_rate", "false_discovery_rate",
            "aggregator"
        ]]
        result_data["display_aggregator"] = result_data["aggregator"].apply(
            lambda x: self.aggregator_map[x])
        result_data = result_data.drop("aggregator", axis=1)

        result_data["label"] = self.label_map[job_hash]
        result_data["dataset"] = dataset
        result_data["class"] = clazz

        return result_data

    def execute(self):
        df = None
        results = sorted(self.collect_results(self.data_dir))
        # confusions = {}
        for result in mapper(self.execute_result, results):
            if df is None:
                df = result
            else:
                df = df.append(result)

        baseline_results = df[df["label"] == "Baseline"]
        other_results = df[df["label"] != "Baseline"]

        groups = other_results[[
            "display_aggregator", "label", "dataset", "class"
        ]].drop_duplicates()

        for group in groups.iterrows():
            group = group[1]
            clazz = group["class"]
            aggregator = group["display_aggregator"]
            dataset = group["dataset"]
            label = group["label"]

            group_baseline_results = baseline_results[
                (baseline_results["dataset"] == dataset)
                & (baseline_results["class"] == clazz)]
            group_baseline_results = group_baseline_results[[
                "recall", "false_discovery_rate"
            ]]
            group_baseline_results = group_baseline_results.groupby(
                "recall").agg({
                    "false_discovery_rate": 'min'
                }).reset_index()
            group_baseline_results = group_baseline_results.sort_values(
                "recall")

            group_results = other_results[
                (other_results["display_aggregator"] == aggregator)
                & (other_results["label"] == label)
                & (other_results["dataset"] == dataset)
                & (other_results["class"] == clazz)]
            group_results = group_results[["recall", "false_discovery_rate"]]
            group_results = group_results.groupby("recall").agg({
                "false_discovery_rate":
                'min'
            }).reset_index()
            group_results = group_results.append(
                {
                    "recall": 0.,
                    "false_discovery_rate": 0.
                }, ignore_index=True)
            group_results = group_results.append(
                {
                    "recall": 1.,
                    "false_discovery_rate": 1.
                }, ignore_index=True)
            group_results = group_results.sort_values("recall")
            recall = group_results["recall"]
            fdr = group_results["false_discovery_rate"]

            auc_fdr = np.trapz(recall, fdr)

            #Plot False-Discovery Rate
            fig, ax = self.visualize(recall, fdr)
            ax.set_xlabel('False Discovery Rate (1 - Precision)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
            ax.plot(group_baseline_results["false_discovery_rate"],
                    group_baseline_results["recall"], "o")
            subtitle = "{} - Class {}, {} Aggregator".format(
                label, clazz, aggregator)

            plt.figtext(.5, .97, subtitle, fontsize=14, ha='center')
            plt.title(
                'True Positive Rate vs. False Discovery Rate (AUC {:1.2f})'.
                format(round(auc_fdr, 3)),
                y=1.17,
                fontsize=16)

            plt.legend([label, "Baseline"],
                       bbox_to_anchor=(0, 1, 1, 0.2),
                       loc="lower left",
                       ncol=2,
                       frameon=False)

            outdir = os.path.join(self.data_dir, "combined", "results", label,
                                  clazz)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(
                outdir, "{}-auc-false-discovery.png".format("_".join(
                    aggregator.split())))
            print(outfile)
            plt.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def visualize(self, tpr, fpr):
        f, ax = plt.subplots()
        ax.plot(fpr, tpr, marker='o')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, max(fpr)])
        return f, ax

    def collect_results(self, directory):
        return glob.glob("{}/**/metrics.csv".format(directory), recursive=True)
