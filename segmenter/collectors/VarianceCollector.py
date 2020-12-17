import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.collectors.BaseCollector import BaseCollector
import glob
import numpy as np
from typing import Dict
from segmenter.helpers.p_tqdm import p_uimap as mapper


class VarianceCollector(BaseCollector):

    results = pd.DataFrame()

    @staticmethod
    def execute_result(result):
        job_hash = result.split("/")[-3]
        result_df = pd.read_csv(result)
        result_df["base_job"] = job_hash
        result_df = result_df[[
            "job", "class", "sample", "squared_difference", "base_job"
        ]]
        return result_df

    def execute(self):
        outfile = os.path.abspath(
            os.path.join(self.data_dir, "..", "variance.csv"))
        if os.path.exists(outfile):
            os.remove(outfile)

        results = glob.glob("{}/../**/variance.csv".format(self.data_dir),
                            recursive=True)

        results = [
            os.path.abspath(r) for r in results
            if os.path.abspath(r) != outfile
        ]

        for result_df in mapper(VarianceCollector.execute_result, results):
            result_df.to_csv(outfile,
                             mode='a',
                             index=False,
                             header=not os.path.exists(outfile))
