import os
import numpy as np
from matplotlib import pyplot as plt
from segmenter.collectors.BaseCollector import BaseCollector
import glob
from segmenter.helpers.p_tqdm import p_uimap as mapper
from sklearn.metrics import confusion_matrix


class ConfusionCollector(BaseCollector):
    def execute_result(self, result):
        name = os.path.basename(result)[:-4]
        if name == "layer_outputs":
            return

        result_data = np.load(result)

        clazz = result.split("/")[-5]
        aggregator_name = result.split("/")[-3]
        threshold = result.split("/")[-2]

        ground_truth = np.load(
            os.path.join(self.ground_truth_dir, os.path.basename(result)))
        class_index = self.job_config["CLASSES"].index(clazz)

        categorical = np.argmax(np.dstack(
            (np.zeros_like(ground_truth["mask"][:, :,
                                                0]), ground_truth["mask"])),
                                axis=2)
        predictions = (result_data["prediction"] * (class_index + 1)).astype(
            categorical.dtype)

        labels = list(range(len(self.labels)))

        result_confusion = confusion_matrix(categorical.flatten(),
                                            predictions.flatten(),
                                            labels=labels)

        false_negatives = result_confusion[class_index + 1, 0]
        result_confusion[0,
                         0] = np.sum(result_confusion[:, 0]) - false_negatives
        result_confusion[1:, 0] = 0
        result_confusion[class_index + 1, 0] = false_negatives

        return clazz, aggregator_name, threshold, result_confusion.astype(
            np.uint64)

    def execute(self):
        self.labels = ["background"] + self.job_config["CLASSES"]
        confusions = {}
        print(self.data_dir)
        results = sorted(self.collect_results(self.data_dir))
        for result in mapper(self.execute_result, results):
            if result is None:
                continue
            clazz = result[0]
            aggregator_name = result[1]
            threshold = result[2]
            key = (clazz, aggregator_name, threshold)
            if key not in confusions:
                confusions[key] = np.zeros(
                    (len(self.labels), len(self.labels)), dtype=np.uint64)
            result_data = result[3]
            confusions[key] += result_data
        for key, result_data in confusions.items():
            clazz = key[0]
            aggregator_name = key[1]
            threshold = key[2]
            outfile = os.path.join(self.data_dir, aggregator_name, threshold,
                                   'confusion.npy')
            np.save(outfile, result_data)

    def collect_results(self, directory):
        return glob.glob("{}/**/*.npz".format(directory), recursive=True)
