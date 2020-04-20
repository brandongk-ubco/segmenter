import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
import glob
import numpy as np


class ActivationVisualizer(BaseVisualizer):
    def execute(self):
        results = self.collect_results(self.data_dir)
        for result in results:
            print(result)
            plot = self.visualize(np.load(result))
            plt.savefig(os.path.join(
                os.path.dirname(result),
                "{}.png".format(os.path.basename(result)[:-4])),
                        dpi=100,
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()

    def collect_results(self, directory):
        return glob.glob("{}/**/activation-*.npz".format(directory),
                         recursive=True)

    def visualize(self, result):
        plot = plt.plot([1], [1])
        return plot
