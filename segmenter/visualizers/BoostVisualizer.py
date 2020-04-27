import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer


class BoostVisualizer(BaseVisualizer):
    def execute(self):
        csv_file = os.path.join(self.data_dir, "boost.csv")
        clazz = self.data_dir.split("/")[-2]
        if not os.path.exists(csv_file):
            print("CSV file does not exist {}".format(csv_file))
            return
        results = pd.read_csv(csv_file)
        plot = results.boxplot(column=['improvement'],
                               by='boost_fold',
                               grid=False)

        title = "Loss Improvement by Number of Boost Folds"
        subtitle = "Class {}".format(clazz)

        fig = plot.get_figure()
        plt.title('')
        fig.suptitle(title, y=1.05, fontsize=14)
        plt.figtext(.5, .96, subtitle, fontsize=12, ha='center')
        plot.set_ylabel('Improvement Over Baseline Loss (%)')
        plot.set_xlabel('Number of Boost Folds')

        outfile = os.path.join(self.data_dir, "boost.png")
        fig.savefig(outfile, dpi=100, bbox_inches='tight', pad_inches=0.5)
        plt.close()
