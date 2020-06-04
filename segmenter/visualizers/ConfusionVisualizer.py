import os
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
from segmenter.helpers.p_tqdm import t_map as mapper
from sklearn.metrics import ConfusionMatrixDisplay


class ConfusionVisualizer(BaseVisualizer):
    job_combined_visualizer = True

    def execute_result(self, result):
        confusion = np.load(result)

        clazz = result.split("/")[-5]
        aggregator_name = result.split("/")[-3]
        threshold = result.split("/")[-2]

        return clazz, aggregator_name, threshold, confusion

    def execute(self):
        self.labels = ["background"] + self.job_config["CLASSES"]

        results = sorted(self.collect_results(self.data_dir))
        confusions = {}
        for result in mapper(self.execute_result, results):
            aggregator_name = result[1]
            threshold = result[2]
            key = (aggregator_name, threshold)
            if key not in confusions:
                confusions[key] = np.zeros(
                    (len(self.labels), len(self.labels)), dtype=np.uint64)
            result_data = result[3]
            confusions[key] += result_data
        for key, result_data in confusions.items():
            aggregator_name = key[0]
            threshold = key[1]
            outdir = os.path.join(self.data_dir, "combined", "results",
                                  aggregator_name, threshold)
            os.makedirs(outdir, exist_ok=True)
            outfile = os.path.join(outdir, 'confusion.png')
            result_data = result_data.astype(np.float64)
            # confusion = np.round(result_data / np.sum(result_data) * 100, 2)
            confusion = np.round(
                result_data /
                (result_data.sum(axis=0) + np.finfo(result_data.dtype).eps) *
                100, 1)
            confusion_matrix_display = ConfusionMatrixDisplay(
                confusion, display_labels=self.labels).plot()
            fig = confusion_matrix_display.figure_
            fig.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.5)
            plt.close()

    def collect_results(self, directory):
        return glob.glob("{}/**/confusion.npy".format(directory),
                         recursive=True)
