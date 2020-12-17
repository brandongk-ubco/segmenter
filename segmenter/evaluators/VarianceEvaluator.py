import os
import numpy as np
from matplotlib import pyplot as plt
from segmenter.evaluators.BaseEvaluator import BaseEvaluator
import glob
from segmenter.helpers.p_tqdm import p_uimap as mapper
import pandas as pd


class VarianceEvaluator(BaseEvaluator):

    results = pd.DataFrame()

    sample_map = {}

    def execute_result(self, result):
        results_df = pd.DataFrame()
        for sample in self.sample_map.keys():
            [clazz, aggregator, threshold, filename] = sample.split("-")
            expected_file = os.path.join(self.base_dir, result, clazz,
                                         "results", aggregator, threshold,
                                         filename)
            if not os.path.isfile(expected_file):
                continue
            this_sample = self.sample_map[sample]
            other_sample = np.load(expected_file)
            squared_difference = np.sum(
                (this_sample - other_sample["raw_prediction"])**2)
            row_df = pd.DataFrame({
                "job": [result],
                "class": [clazz],
                "sample": [filename[:-4]],
                "squared_difference": [squared_difference]
            })
            results_df = results_df.append(row_df, ignore_index=True)
        return results_df

    def populate_sample(self, sample):
        clazz = sample.split("/")[-5]
        aggregator = sample.split("/")[-3]
        threshold = sample.split("/")[-2]
        filename = sample.split("/")[-1]
        key = "-".join([clazz, aggregator, threshold, filename])
        prediction = np.load(sample)
        return key, prediction["raw_prediction"]

    def populate_samples(self):
        samples = glob.glob("{}/**/*.npz".format(self.data_dir),
                            recursive=True)
        for key, prediction in mapper(self.populate_sample, samples):
            self.sample_map[key] = prediction

    def execute(self):
        outdir = os.path.join(self.data_dir, "results")
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "variance.csv")
        if os.path.exists(outfile):
            return

        job_configs = sorted(self.collect_results(self.data_dir))
        self.populate_samples()
        for results_df in mapper(self.execute_result, job_configs):
            self.results = self.results.append(results_df, ignore_index=True)
        self.results.to_csv(outfile)

    def collect_results(self, directory):
        self.base_dir = os.path.abspath(os.path.join(directory, ".."))
        job_hash = os.path.basename(os.path.normpath(self.data_dir))
        job_configs = [
            d for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]
        job_configs = sorted(job_configs)
        job_configs = job_configs[job_configs.index(job_hash) + 1:]
        return job_configs
