import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from segmenter.aggregators import Aggregators
import glob
import numpy as np


class PredictionVisualizer(BaseVisualizer):
    def execute(self):
        results = self.collect_results(self.data_dir)
        for result in results:
            print(result)
            r = np.load(result)

            clazz = result.split("/")[-5]
            aggregator_name = result.split("/")[-3]
            threshold = result.split("/")[-2]
            aggregator = Aggregators.get(aggregator_name)
            name = os.path.basename(result)[11:-4]
            title = "Predictions for {}".format(name)
            subtitle = "Class {}, {} Aggregator with Threshold {}".format(
                clazz, aggregator.display_name(), threshold)
            plot = self.visualize(r["image"], r["mask"], r["prediction"])
            plot.suptitle(title, y=1.05, fontsize=16)
            plt.figtext(.5, .96, subtitle, fontsize=14, ha='center')
            plt.savefig(os.path.join(os.path.dirname(result),
                                     "{}.png".format(name)),
                        dpi=100,
                        bbox_inches='tight',
                        pad_inches=0.5)
            plt.close()

    def collect_results(self, directory):
        return glob.glob("{}/**/prediction-*.npz".format(directory),
                         recursive=True)

    def visualize(self, image, mask, prediction):
        boolean_mask = mask.astype(bool)
        boolean_prediction = prediction.astype(bool)

        # False Positives
        fp = np.zeros(image.shape, dtype=image.dtype)
        fp[np.logical_and(np.logical_xor(boolean_mask, boolean_prediction),
                          boolean_prediction)] = 1.

        # True Positives
        tp = np.zeros(image.shape, dtype=image.dtype)
        tp[np.logical_and(boolean_mask, boolean_prediction)] = 1.

        # False Negatives
        fn = np.zeros(image.shape, dtype=image.dtype)
        fn[np.logical_and(np.logical_xor(boolean_mask, boolean_prediction),
                          np.logical_not(boolean_prediction))] = 1.

        highlighted_mask = np.dstack((fp, tp, fn))

        highlighted_image = np.dstack(
            (image.copy(), image.copy(), image.copy()))

        highlighted_image[highlighted_mask == 1] = 1.

        intr = np.sum(np.logical_and(boolean_prediction, boolean_mask))
        union = np.sum(np.logical_or(boolean_prediction, boolean_mask))
        iou = intr / union

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

        fig.set_size_inches(10, 5)

        ax1.imshow(image, cmap='gray')
        ax1.axis('off')
        ax1.set_title('Image')

        ax2.imshow(mask, cmap='gray')
        ax2.axis('off')
        ax2.set_title('Mask')

        ax3.imshow(highlighted_mask, cmap='gray')
        ax3.set_title("Predicted Mask (IOU {:.2f})".format(iou))
        ax3.axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)

        return fig
