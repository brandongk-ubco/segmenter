import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
from segmenter.helpers.p_tqdm import t_map as mapper
from sklearn.metrics import ConfusionMatrixDisplay


class CombinedLayerOutputVisualizer(BaseVisualizer):

    full_combined_visualizer = True
    bins = np.linspace(-10, 10, num=2001)

    def execute_result(self, result):
        result_data = pd.read_csv(result)

        job_hash = result.split("/")[-4]
        category = "Baseline" if self.label_map[
            job_hash] == "Baseline" else "Full-Domain"
        result_data["category"] = category
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
        for layer_type in df["layer_type"].unique():
            layer_type_results = df[df["layer_type"] == layer_type]
            layer_type_results = layer_type_results.drop("layer_type", axis=1)
            layer_type_results = layer_type_results.drop("fold", axis=1)
            layer_type_results = layer_type_results.drop("boost_fold", axis=1)

            layer_type_results = layer_type_results.groupby("category").agg(
                'sum').reset_index()
            for category in layer_type_results["category"].unique():
                category_results = layer_type_results[
                    layer_type_results["category"] == category]
                category_results = category_results.drop("category", axis=1)
                category_results = category_results.to_numpy()[0]

                weights = 100 * category_results / np.sum(category_results)

                bins = self.bins[:len(weights)]
                mean = np.sum(np.multiply(category_results,
                                          bins)) / np.sum(category_results)
                std = np.sum(np.multiply(category_results, (bins - mean)**
                                         2)) / np.sum(category_results)

                fig = plt.figure()
                plt.hist(bins, self.bins, weights=weights)

                percentile = np.percentile(weights, 99.9)
                plt.ylim([0, percentile])

                title = "Output Histogram for {} {} layers".format(
                    category, layer_type)
                plt.ylabel("Frequency (%): Peak {:1.2f}% at {:1.2f}".format(
                    np.max(weights), self.bins[np.argmax(weights)]))
                used_bins = weights > 0.01
                subtitle = "Frequency Concentration:  {:1.2f}% in width {:1.2f}".format(
                    np.sum(weights[used_bins]),
                    max(bins[used_bins]) - min(bins[used_bins]))
                plt.xlabel(
                    "Output Value: Mean {:1.2f}, St. Dev. {:1.2f}".format(
                        mean, std))
                plt.title('')
                fig.suptitle(title, y=1.00, fontsize=14)
                plt.figtext(.5, .91, subtitle, fontsize=12, ha='center')
                outdir = os.path.join(self.data_dir, "combined", "results")
                os.makedirs(outdir, exist_ok=True)
                outfile = os.path.join(
                    outdir,
                    "layer-output-{}-{}.png".format(category, layer_type))
                print(outfile)
                plt.savefig(outfile,
                            dpi=150,
                            bbox_inches='tight',
                            pad_inches=0.5)
                plt.close()

    def collect_results(self, directory):
        return glob.glob("{}/**/layer-outputs.csv".format(directory),
                         recursive=True)
