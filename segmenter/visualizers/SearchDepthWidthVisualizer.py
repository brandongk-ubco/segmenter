import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import seaborn as sns


class SearchDepthWidthVisualizer(BaseVisualizer):
    def plot(self, results):
        results = results[[
            "class", "model_filters", "model_layers", "val_loss"
        ]]
        results = results.rename(columns={
            "model_filters": "filters",
            "model_layers": "layers"
        })
        results = results.groupby(["class", 'filters',
                                   'layers']).min().reset_index()
        for clazz in results["class"].unique().tolist():
            clazz_results = results[results["class"] == clazz]
            clazz_results = clazz_results.pivot("filters", "layers",
                                                "val_loss")

            sns.set(rc={'figure.figsize': (11, 2.5)})
            plot = sns.heatmap(clazz_results,
                               annot=True,
                               fmt='.4g',
                               linewidths=.5)
            fig = plot.get_figure()
            plt.title("Class %s" % clazz)

            outfile = os.path.join(self.data_dir,
                                   "class_%s_size_heatmap.png" % clazz)
            fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def execute(self):
        csv_file = os.path.join(self.data_dir, "train_results.csv")
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        results = pd.read_csv(csv_file)
        self.plot(results.copy())
