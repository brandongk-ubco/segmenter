import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer


class BoostVisualizer(BaseVisualizer):
    def boxplot(self, clazz, results):
        results = results[["loss", "fold", "boost_fold"]]
        results = results.groupby(['fold', 'boost_fold']).min().reset_index()
        results = results.sort_values(["fold", "boost_fold"])
        results["loss"] = pd.to_numeric(results["loss"], errors='coerce')
        results["baseline"] = results.apply(lambda x: results[
            (results["fold"] == x["fold"]) &
            (results["boost_fold"] == 0)].iloc[0].loss,
                                            axis=1)
        results["improvement"] = results.apply(
            lambda x: 100 * (x["baseline"] - x["loss"]) / x["baseline"],
            axis=1)
        results = results[results["boost_fold"] > 0]

        plot = results.boxplot(column=['improvement'],
                               by='boost_fold',
                               grid=False)

        title = "Loss Improvement by Number of Boost Folds"
        subtitle = "{} - Class {}".format(self.label, clazz)

        fig = plot.get_figure()
        plt.title('')
        fig.suptitle(title, y=1.05, fontsize=14)
        plt.figtext(.5, .96, subtitle, fontsize=12, ha='center')
        plot.set_ylabel('Improvement Over Baseline Loss (%)')
        plot.set_xlabel('Number of Boost Folds')

        outfile = os.path.join(self.data_dir, "boost_boxplot.png")
        fig.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def lineplot(self, clazz, results):
        results = results[["epoch", "loss", "boost_fold"]]
        results = results.rename(columns={'boost_fold': 'Boost Fold'})

        results["loss"] = pd.to_numeric(results["loss"], errors='coerce')
        results = results[results['loss'].notna()]
        results = results.groupby(["epoch", 'Boost Fold']).mean().unstack()
        plot = results["loss"].plot.line()

        title = "Mean Validation Loss by Training Epoch"
        subtitle = "{} - Class {}".format(self.label, clazz)

        fig = plot.get_figure()
        plt.title('')
        fig.suptitle(title, y=1.05, fontsize=14)
        plt.figtext(.5, .96, subtitle, fontsize=12, ha='center')
        plt.ylabel('Validation Loss')
        plt.xlabel('Training Epoch')

        outfile = os.path.join(self.data_dir, "boost_lineplot.png")
        fig.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.5)
        plt.close()

    def execute(self):
        csv_file = os.path.join(self.data_dir, "train_results.csv")
        clazz = self.data_dir.split("/")[-2]
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        results = pd.read_csv(csv_file)
        self.boxplot(clazz, results.copy())
        self.lineplot(clazz, results.copy())
