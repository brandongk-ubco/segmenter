import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from segmenter.visualizers.BaseVisualizer import BaseVisualizer
from segmenter.aggregators import Aggregators
import glob
import numpy as np
from matplotlib.lines import Line2D
from segmenter.helpers.p_tqdm import p_map as mapper


class PredictionVisualizer(BaseVisualizer):
    def execute_result(self, result):
        name = os.path.basename(result)[:-4]
        if name == "layer_outputs":
            return

        outfile = os.path.join(os.path.dirname(result), "{}.png".format(name))
        # if os.path.exists(outfile):
        #     return

        r = np.load(result)

        clazz = result.split("/")[-5]
        aggregator_name = result.split("/")[-3]
        threshold = result.split("/")[-2]
        aggregator = Aggregators.get(aggregator_name)(self.job_config)

        plot, iou = self.visualize(r["image"], r["mask"], r["prediction"])
        title = "Predictions for {} (IOU {:.2f})".format(name, iou)
        subtitle = "{} - Class {}, {} Aggregator with Threshold {}".format(
            self.label, clazz, aggregator.display_name(), threshold)
        if aggregator.display_name() == "Dummy":
            subtitle = "{} - Class {} with Threshold {}".format(
                self.label, clazz, threshold)
        # TODO: These values need to be tweaked based on image size.
        # Try to find a way to calculate these.
        if "/kits19/" in os.path.dirname(result):
            plot.suptitle(title, y=0.96, fontsize=16)
            plt.figtext(.5, 0.92, subtitle, fontsize=14, ha='center')
        elif "/severstal/" in os.path.dirname(result):
            plot.suptitle(title, y=1.35, fontsize=16)
            plt.figtext(.5, 1.12, subtitle, fontsize=14, ha='center')
        else:
            raise ValueError("Couldn't determine dataset.")
        plt.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def execute(self):
        print(self.data_dir)
        results = sorted(self.collect_results(self.data_dir))
        mapper(self.execute_result, results)

    def collect_results(self, directory):
        return glob.glob("{}/**/*.npz".format(directory), recursive=True)

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

        alphas = 1 - np.logical_or(np.logical_or(fp, tp), fn).astype(
            image.dtype)

        highlighted_mask = np.dstack((fp, tp, fn))

        highlighted_image = np.dstack(
            (image.copy() * alphas, image.copy() * alphas,
             image.copy() * alphas))

        highlighted_image[highlighted_mask == 1] = 1.

        intr = np.sum(np.logical_and(boolean_prediction, boolean_mask))
        union = np.sum(np.logical_or(boolean_prediction, boolean_mask))
        iou = intr / union

        legend = [
            Line2D(
                [0],
                [0],
                color='r',
                lw=4,
            ),
            Line2D([0], [0], color='g', lw=4),
            Line2D([0], [0], color='b', lw=4)
        ]
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', interpolation='none')
        ax.imshow(highlighted_image, alpha=0.5, interpolation='none')
        fig.set_size_inches(11,
                            11 * boolean_mask.shape[0] / boolean_mask.shape[1])
        ax.axis('off')
        plt.legend(legend,
                   ['False Positive', 'True Positive', 'False Negative'],
                   bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   frameon=False,
                   ncol=3)

        return fig, iou
