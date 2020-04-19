import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def results_from_path(path):
    roc_dir = os.path.abspath("./test_fixtures/roc")
    methods = sorted([m for m in os.listdir(roc_dir) if os.path.isdir(os.path.join(roc_dir, m))])
    results = {}
    for method in methods:
        method_dir = os.path.join(roc_dir, method)
        samples = sorted([os.path.join(method_dir, o) for o in os.listdir(method_dir) if os.path.isdir(os.path.join(method_dir, o))])
        method_results = []
        for sample in samples:
            with open(os.path.join(sample, "results.json"), "r") as sample_file:
                sample_json = json.load(sample_file)
                method_results.append(sample_json)
        results[method] = method_results
    return results

def compile_results(results):
    tpr = [0.0]
    fpr = [0.0]
    for result in results:
        tpr.append(round(result['recall'], 2))
        fpr.append(round((1 - result['specificity']), 2))
    df = pd.DataFrame({"tpr": tpr, "fpr": fpr})
    df = df.sort_values(['tpr', 'fpr'], ascending=True).drop_duplicates(['fpr'])
    tpr = np.array(df["tpr"])
    fpr = np.array(df["fpr"])
    auc = max(tpr) * (1 - max(fpr)) + np.trapz(tpr, fpr)
    return tpr, fpr, auc

