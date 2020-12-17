import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from segmenter.config import config_from_dir
from deepdiff import DeepDiff


class BestPairVisualizer(BaseVisualizer):

    configs = {}

    def read_config(self, job_hash):
        if job_hash not in self.configs:
            config, hash = config_from_dir(
                os.path.join(self.data_dir, job_hash))
            assert (job_hash == hash)
            self.configs[job_hash] = config
        return self.configs[job_hash]

    def execute(self):
        pair_files = [
            os.path.join(self.data_dir, d) for d in os.listdir(self.data_dir)
            if d.endswith("best_pairs.csv")
        ]

        result_df = pd.DataFrame(columns=['class'], dtype=str)
        count_df = pd.DataFrame(columns=['class'], dtype=str)

        for pair_file in pair_files:
            pair_df = pd.read_csv(pair_file)
            for i, row in pair_df.iterrows():
                job1_config = self.read_config(row["job"])
                job2_config = self.read_config(row["base_job"])
                diff = DeepDiff(job1_config, job2_config, ignore_order=True)
                count = len([
                    (k, v) for k, v in diff["values_changed"].items()
                    if "_".join(k.split("'")[1:-1:2]) != "MAX_TRAIN_SIZE"
                ])
                if count == 0:
                    continue
                count_df = count_df.append(
                    {
                        "count": count,
                        "difference": row["squared_difference"],
                        "class": row["class"]
                    },
                    ignore_index=True)
                for k, v in diff["values_changed"].items():
                    key = "_".join(k.split("'")[1:-1:2])
                    if key == "MAX_TRAIN_SIZE":
                        continue
                    key = key.replace("MODEL_", "")
                    result_df = result_df.append(
                        {
                            "key": key,
                            "val_1": min(v["old_value"], v["new_value"]),
                            "val_2": max(v["old_value"], v["new_value"]),
                            "difference": row["squared_difference"],
                            "class": row["class"]
                        },
                        ignore_index=True)

        result_mean_df = result_df.groupby(["class", "key", "val_1",
                                            "val_2"]).mean().reset_index()

        result_count_df = result_df.groupby(["class", "key", "val_1",
                                             "val_2"]).size().reset_index()
        result_combined_df = result_mean_df.merge(result_count_df).sort_values(
            ["class", "key", "val_1", "val_2"])
        result_combined_df = result_combined_df.rename(columns={0: "count"})
        for clazz in result_combined_df["class"].unique():
            outfile = os.path.join(self.data_dir,
                                   "%s_diversity_table.tex" % clazz)
            with open(outfile, "w") as outbuf:
                clazz_df = result_combined_df[result_combined_df["class"] ==
                                              clazz]
                clazz_df = clazz_df[[
                    "key", "val_1", "val_2", "difference", "count"
                ]]
                clazz_df.to_latex(buf=outbuf,
                                  index=False,
                                  float_format="{:0.4g}".format)

        count_df = count_df[count_df["count"] > 0].copy()
        count_mean_df = count_df.groupby(["class",
                                          "count"]).mean().reset_index()

        for clazz in count_mean_df["class"].unique():
            outfile = os.path.join(
                self.data_dir,
                "%s_diversity_count_table.tex" % str(int(clazz)))
            with open(outfile, "w") as outbuf:
                clazz_df = count_mean_df[count_mean_df["class"] == clazz]
                clazz_df = clazz_df[["count", "difference"]]
                clazz_df = clazz_df.sort_values("count", ascending=True)
                clazz_df.to_latex(buf=outbuf,
                                  index=False,
                                  float_format="{:0.4g}".format)
