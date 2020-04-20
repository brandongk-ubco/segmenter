import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer


class AUCVisualizer(BaseVisualizer):
    def execute(self):
        results = self.collect_results(self.data_dir)
        for method, result in results.items():
            tpr, fpr, auc = self.compile_results(result)
            plot = self.visualize(method, tpr, fpr, auc)
            plot.savefig(os.path.join(self.data_dir, "{}.png".format(method)))
            plot.close()

    def collect_results(self, directory):
        methods = sorted([
            m for m in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, m))
        ])
        results = {}
        for method in methods:
            method_dir = os.path.join(directory, method)
            samples = sorted([
                os.path.join(method_dir, o) for o in os.listdir(method_dir)
                if os.path.isdir(os.path.join(method_dir, o))
            ])
            method_results = []
            for sample in samples:
                with open(os.path.join(sample, "results.json"),
                          "r") as sample_file:
                    sample_json = json.load(sample_file)
                    method_results.append(sample_json)
            results[method] = method_results
        return results

    def compile_results(self, results):
        tpr = [0.0, 1.0]
        fpr = [0.0, 1.0]
        for result in results:
            tpr.append(round(result['recall'], 2))
            fpr.append(round((1 - result['specificity']), 3))
        df = pd.DataFrame({"tpr": tpr, "fpr": fpr})
        df = df.sort_values(['tpr', 'fpr'],
                            ascending=[True,
                                       True]).drop_duplicates(['fpr'],
                                                              keep="last")
        tpr = np.array(df["tpr"])
        fpr = np.array(df["fpr"])
        auc = max(tpr) * (1 - max(fpr)) + np.trapz(tpr, fpr)
        return tpr, fpr, auc

    def visualize(self, method, tpr, fpr, auc):
        plt.plot(fpr, tpr, marker='o')
        plt.title(
            'Receiver Operating Characteristic Curve for {} (AUC = {})'.format(
                method, round(auc, 3)))
        plt.ylim([0, 1])
        plt.xlim([0, max(fpr)])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        return plt
