import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
from segmenter.helpers.p_tqdm import t_map as mapper
from sklearn.metrics import ConfusionMatrixDisplay
from segmenter.aggregators import Aggregators


class ConfusionVisualizer(BaseVisualizer):
    job_combined_visualizer = True

    aggregator_map = dict(
        [("dummy", "")] +
        [(a.name(), a.display_name())
         for a in [Aggregators.get(a)(None) for a in Aggregators.choices()]])

    def execute_result(self, result):
        confusion = np.load(result)

        clazz = result.split("/")[-5]
        aggregator_name = result.split("/")[-3]
        threshold = result.split("/")[-2]

        return clazz, aggregator_name, threshold, confusion

    def executre_metrics(self, result):
        result_data = pd.read_csv(result)

        job_hash = result.split("/")[-4]
        dataset = result.split("/")[-5]
        clazz = result.split("/")[-3]
        result_data["label"] = self.label_map[job_hash]
        result_data["dataset"] = dataset
        result_data["class"] = clazz
        return result_data

    def find_best_thresholds(self):
        df = None
        metrics_files = glob.glob("{}/**/metrics.csv".format(self.data_dir),
                                  recursive=True)
        results = sorted(metrics_files)
        # confusions = {}
        df = None
        for result in mapper(self.executre_metrics, results):
            if df is None:
                df = result
            else:
                df = df.append(result)

        mean_results = df.groupby(
            ["label", "dataset", "aggregator", "threshold", "class"]).agg({
                "f1-score":
                "mean"
            }).reset_index()

        best_results = mean_results.groupby(
            ["label", "dataset", "aggregator", "class"]).agg({
                "f1-score": "max"
            }).reset_index()

        best_results = pd.merge(best_results,
                                mean_results,
                                on=list(best_results.columns),
                                how='inner')

        return best_results

    def execute(self):
        self.labels = ["background"] + self.job_config["CLASSES"]
        bt = self.find_best_thresholds()

        for aggregator in bt["aggregator"].unique():
            agregator_results = bt[bt["aggregator"] == aggregator]

            outdir = os.path.join(self.data_dir, "combined", "results",
                                  aggregator)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, 'confusion.png')
            confusion = np.zeros((len(self.labels), len(self.labels)),
                                 dtype=np.uint64)

            for clazz in agregator_results["class"].unique():
                best_threshold = bt[
                    (bt["aggregator"] == aggregator)
                    & (bt["class"] == clazz)]["threshold"].iloc[0]
                confusion_file = os.path.join(self.data_dir, str(clazz),
                                              "results", aggregator,
                                              "{:.2f}".format(best_threshold),
                                              "confusion.npy")
                result_data = np.load(confusion_file)
                confusion += result_data

            confusion = confusion.astype(np.float64)
            confusion = np.round(
                confusion /
                (confusion.sum(axis=0) + np.finfo(confusion.dtype).eps) * 100,
                1)
            confusion_matrix_display = ConfusionMatrixDisplay(
                confusion, display_labels=self.labels).plot()

            for row in confusion_matrix_display.text_:
                for item in row:
                    item.set_fontsize(20)
            subtitle = "{}".format(self.label)

            ax = confusion_matrix_display.ax_
            for item in ([ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)
            fig = confusion_matrix_display.figure_

            plt.title('')
            fig.suptitle('Confusion Matrix', y=1, fontsize=14)
            plt.figtext(.5, .91, subtitle, fontsize=12, ha='center')

            fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
            plt.close()
