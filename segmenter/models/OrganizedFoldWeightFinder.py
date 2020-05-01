from segmenter.models.FoldWeightFinder import FoldWeightFinder
import os


class OrganizedFoldWeightFinder(FoldWeightFinder):
    def __init__(self, directory):
        self.directory = os.path.join(directory, "results", "weights")
        fold_weights = [
            os.path.join(self.directory, d) for d in os.listdir(self.directory)
        ]
        fold_names = [f.split("/")[-1][:-3] for f in fold_weights]
        self.folds = dict(zip(fold_names, fold_weights))

    def get(self, fold_name):
        assert fold_name in self.folds, "Fold {} does not exist in loader".format(
            fold_name)
        return self.folds[fold_name]

    def keys(self):
        return self.folds.keys()
