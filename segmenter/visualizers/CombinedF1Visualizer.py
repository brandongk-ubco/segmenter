import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
from segmenter.helpers.p_tqdm import t_map as mapper
from segmenter.aggregators import Aggregators
import seaborn as sns


class CombinedF1Visualizer(BaseVisualizer):

    full_combined_visualizer = True

    aggregator_map = dict(
        [("dummy", "")] +
        [(a.name(), a.display_name())
         for a in [Aggregators.get(a)(None) for a in Aggregators.choices()]])

    def execute_result(self, result):
        result_data = pd.read_csv(result)

        job_hash = result.split("/")[-4]
        dataset = result.split("/")[-5]
        clazz = result.split("/")[-3]
        result_data["label"] = self.label_map[job_hash]
        result_data["dataset"] = dataset
        result_data["class"] = clazz
        return result_data

    def execute(self):
        # self.labels = ["background"] + self.job_config["CLASSES"]
        df = None
        results = sorted(self.collect_results(self.data_dir))
        # confusions = {}
        for result in mapper(self.execute_result, results):
            if df is None:
                df = result
            else:
                df = df.append(result)
        df["display_aggregator"] = df["aggregator"].apply(
            lambda x: self.aggregator_map[x])
        df = df.drop("aggregator", axis=1)

        mean_results = df.groupby(
            ["label", "dataset", "display_aggregator", "threshold",
             "class"]).agg({
                 "dice": "mean"
             }).reset_index()

        best_results = mean_results.groupby(
            ["label", "dataset", "display_aggregator", "class"]).agg({
                "dice":
                "max"
            }).reset_index()

        best_results = pd.merge(best_results,
                                mean_results,
                                on=list(best_results.columns),
                                how='inner')

        join_columns = list(best_results.columns)
        join_columns.remove("dice")

        filtered_results = pd.merge(best_results,
                                    df,
                                    on=join_columns,
                                    how='inner')
        filtered_results["dice"] = filtered_results["dice_y"]
        filtered_results = filtered_results.drop("dice_x", axis=1)
        filtered_results = filtered_results.drop("dice_y", axis=1)
        baseline_results = df[df["label"] == "Baseline"]

        sns.set(rc={'figure.figsize': (11, 4)})
        for aggregator in filtered_results["display_aggregator"].unique():
            if aggregator == "":
                continue
            aggregator_results = filtered_results[
                filtered_results["display_aggregator"] == aggregator]
            comparable_results = pd.concat(
                [aggregator_results, baseline_results])
            plot = sns.boxplot(x='class',
                               y='dice',
                               data=comparable_results,
                               hue='label')
            fig = plot.get_figure()

            plt.legend(bbox_to_anchor=(0, 1, 1, 0.2),
                       loc="lower left",
                       ncol=len(comparable_results["label"].unique()),
                       frameon=False)
            plt.title('')
            fig.suptitle('F1-Score by Model and Class', y=1.02, fontsize=14)
            plt.xlabel("Class")
            plt.ylabel("F1-Score")

            outdir = os.path.join(self.data_dir, "combined", "results")
            os.makedirs(outdir, exist_ok=True)

            outfile = os.path.join(
                outdir,
                "{}-f1-score.png".format(aggregator.replace(" ", "_").lower()))
            fig.savefig(outfile, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def collect_results(self, directory):
        return glob.glob("{}/**/instance-metrics.csv".format(directory),
                         recursive=True)
