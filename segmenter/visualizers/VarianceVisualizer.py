import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from statistics import harmonic_mean


class VarianceVisualizer(BaseVisualizer):
    def execute(self):
        variance_df = pd.read_csv(os.path.join(self.data_dir, "variance.csv"))
        variance_df = variance_df[[
            "job", "class", "squared_difference", "base_job"
        ]]
        variance_df = variance_df.groupby(["job", "class",
                                           "base_job"]).mean().reset_index()

        train_results_df = pd.read_csv(
            os.path.join(self.data_dir, "train_results.csv"))
        train_results_df = train_results_df[["job_hash", "class", "val_loss"]]
        train_results_df = train_results_df.groupby(["job_hash", "class"
                                                     ]).min().reset_index()
        worst_results = train_results_df[["class", "val_loss"
                                          ]].groupby(["class"
                                                      ]).max().reset_index()
        worst_results = worst_results.set_index(["class"])
        train_results_df["improvement"] = train_results_df.apply(
            lambda r: (worst_results.loc[r["class"]] - r["val_loss"]
                       ) / worst_results.loc[r["class"]],
            axis=1)
        train_results_df = train_results_df[[
            "job_hash", "class", "improvement"
        ]].set_index(["job_hash", "class"])

        variance_df["mean_improvement"] = variance_df.apply(
            lambda r: harmonic_mean([
                train_results_df.loc[(r["base_job"], r["class"])][
                    "improvement"], train_results_df.loc[
                        (r["job"], r["class"])]["improvement"]
            ]),
            axis=1)

        for clazz in variance_df["class"].unique():
            clazz_df = variance_df[variance_df["class"] == clazz]
            plot = clazz_df.plot.hexbin("squared_difference",
                                        "mean_improvement",
                                        gridsize=100)
            fig = plot.get_figure()
            plt.title("Class %s" % clazz)
            plt.ylabel("Harmonic Mean Loss improvement over baseline (%)")
            plt.xlabel("Squared Difference")
            outfile = os.path.join(self.data_dir,
                                   "%s_variance_improvement.png" % clazz)
            fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
            plt.close()

        for clazz in variance_df["class"].unique():
            clazz_df = variance_df[variance_df["class"] == clazz]
            clazz_df = clazz_df.nlargest(100, ["mean_improvement"])
            plot = clazz_df.plot.scatter("squared_difference",
                                         "mean_improvement")
            fig = plot.get_figure()
            plt.title("Class %s" % clazz)
            plt.ylabel("Harmonic Mean Loss improvement over baseline (%)")
            plt.xlabel("Squared Difference")
            outfile = os.path.join(self.data_dir,
                                   "%s_top_variance_improvement.png" % clazz)
            fig.savefig(outfile, dpi=150, bbox_inches='tight', pad_inches=0.5)
            plt.close()

            clazz_df.to_csv(
                os.path.join(self.data_dir, "%s_best_pairs.csv" % clazz))
