import os
import json
from numpy.lib.financial import rate
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from statistics import harmonic_mean
from math import sqrt
from segmenter.config import config_from_dir


class EnsembleVisualizer(BaseVisualizer):

    configs = {}

    def read_config(self, job_hash):
        if job_hash not in self.configs:
            config, hash = config_from_dir(
                os.path.join(self.data_dir, job_hash))
            assert (job_hash == hash)
            self.configs[job_hash] = config
        return self.configs[job_hash]

    def execute(self):
        variance_loss_file = os.path.join(self.data_dir,
                                          "variance_and_loss.csv")
        if not os.path.exists(variance_loss_file):
            variance_df = pd.read_csv(
                os.path.join(self.data_dir, "variance.csv"))
            variance_df = variance_df[[
                "job", "class", "squared_difference", "base_job"
            ]]
            variance_df = variance_df.groupby(["job", "class", "base_job"
                                               ]).mean().reset_index()

            train_results_df = pd.read_csv(
                os.path.join(self.data_dir, "train_results.csv"))
            train_results_df = train_results_df[[
                "job_hash", "class", "val_loss"
            ]]
            train_results_df = train_results_df.groupby(
                ["job_hash", "class"]).min().reset_index()
            worst_results = train_results_df[["class", "val_loss"]].groupby(
                ["class"]).max().reset_index()
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
            variance_df.to_csv(variance_loss_file)
        else:
            variance_df = pd.read_csv(os.path.join(variance_loss_file))

        ensembles_df = pd.DataFrame()
        for clazz in variance_df["class"].unique():
            clazz_df = variance_df[variance_df["class"] == clazz].copy()
            clazz_df = clazz_df.nlargest(100, ["mean_improvement"])

            normalize_clazz_df = clazz_df.copy()
            normalize_clazz_df[
                "normalized_squared_difference"] = normalize_clazz_df[
                    "squared_difference"] - normalize_clazz_df[
                        "squared_difference"].min()
            normalize_clazz_df[
                "normalized_squared_difference"] = normalize_clazz_df[
                    "normalized_squared_difference"] / normalize_clazz_df[
                        "normalized_squared_difference"].max()

            normalize_clazz_df[
                "normalized_mean_improvement"] = normalize_clazz_df[
                    "mean_improvement"] - normalize_clazz_df[
                        "mean_improvement"].min()
            normalize_clazz_df[
                "normalized_mean_improvement"] = normalize_clazz_df[
                    "normalized_mean_improvement"] / normalize_clazz_df[
                        "normalized_mean_improvement"].max()

            normalize_clazz_df["distance"] = normalize_clazz_df.apply(
                lambda r: harmonic_mean([
                    r["normalized_squared_difference"], r[
                        "normalized_mean_improvement"]
                ]),
                axis=1)

            normalize_clazz_df = normalize_clazz_df.sort_values(
                "distance", ascending=False)

            job_hashes = set()
            selected = set()
            for i, r in normalize_clazz_df.iterrows():
                job_hashes.add(r["job"])
                job_hashes.add(r["base_job"])
                selected.add((r["squared_difference"], r["mean_improvement"]))

                if len(job_hashes) >= 10:
                    break

            ensemble_df = pd.DataFrame()

            for hash in job_hashes:
                job_config = self.read_config(hash)
                ensemble_df = ensemble_df.append(
                    {
                        "CLASS": clazz,
                        "JOB_HASH": hash,
                        "ACTIVATION": job_config["MODEL"]["ACTIVATION"],
                        "FILTERS": job_config["MODEL"]["FILTERS"],
                        "LAYERS": job_config["MODEL"]["LAYERS"],
                        "L1_REG": job_config["L1_REG"],
                    },
                    ignore_index=True)

            ensembles_df = ensembles_df.append(ensemble_df)

            outfile = os.path.join(self.data_dir,
                                   "%s_ensemble.tex" % str(int(clazz)))
            with open(outfile, "w") as outbuf:
                ensemble_df[["ACTIVATION", "FILTERS", "LAYERS",
                             "L1_REG"]].to_latex(buf=outbuf,
                                                 index=False,
                                                 float_format="{:0.4g}".format)

            plot = normalize_clazz_df.plot.scatter("squared_difference",
                                                   "mean_improvement")
            plt.scatter([s[0] for s in selected], [s[1] for s in selected],
                        color="orange")
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
        ensembles_df.to_csv(os.path.join(self.data_dir, "ensembles.csv"),
                            index=False)
