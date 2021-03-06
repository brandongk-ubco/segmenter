from segmenter.models.FoldWeightFinder import FoldWeightFinder
import os


class BestFoldWeightFinder(FoldWeightFinder):
    def __init__(self, directory):
        self.directory = directory

    @staticmethod
    def find_best_in_directory(folder: str):
        if not os.path.isdir(folder):
            return None
        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")
        ]
        if len(files) == 0:
            return None
        return min(files, key=lambda x: float(x.split("-")[1][:-3]))

    def get(self, fold_name):
        fold_directory = os.path.join(self.directory, fold_name)
        assert os.path.exists(
            fold_directory), "Fold {} does not exist in loader".format(
                fold_name)

        return BestFoldWeightFinder.find_best_in_directory(
            os.path.join(self.directory, fold_name))

    def keys(self):
        return [
            d for d in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, d))
            and d not in ["results"]
        ]
