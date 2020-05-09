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
from segmenter.helpers.p_tqdm import p_map


class PredictionVisualizer(BaseVisualizer):
    def execute_result(self, result):
        name = os.path.basename(result)[:-4]
        if name == "layer_outputs":
            return

        outfile = os.path.join(os.path.dirname(result), "{}.jpg".format(name))
        if os.path.exists(outfile):
            return

        r = np.load(result)

        clazz = result.split("/")[-5]
        aggregator_name = result.split("/")[-3]
        threshold = result.split("/")[-2]
        aggregator = Aggregators.get(aggregator_name)(self.job_config)

        plot, iou = self.visualize(r["image"], r["mask"], r["prediction"])
        title = "Predictions for {} (IOU {:.2f})".format(name, iou)
        subtitle = "{} - Class {}, {} Aggregator with Threshold {}".format(
            self.label, clazz, aggregator.display_name(), threshold)

        plot.suptitle(title, y=1.35, fontsize=16)
        plt.figtext(.5, 1.12, subtitle, fontsize=14, ha='center')
        plt.savefig(outfile,
                    dpi=70,
                    bbox_inches='tight',
                    pad_inches=0.5,
                    quality=90,
                    optimize=True)
        plt.close()

    def execute(self):
        print(self.data_dir)
        results = sorted(self.collect_results(self.data_dir))
        p_map(self.execute_result, results)

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

        highlighted_mask = np.dstack((fp, tp, fn))

        highlighted_image = np.dstack(
            (image.copy(), image.copy(), image.copy()))

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

        plot = plt.imshow(highlighted_image, cmap='gray')
        fig = plot.get_figure()
        fig.set_size_inches(11,
                            11 * boolean_mask.shape[0] / boolean_mask.shape[1])
        plt.axis('off')
        plt.legend(legend,
                   ['False Positive', 'True Positive', 'False Negative'],
                   bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   frameon=False,
                   ncol=3)

        return fig, iou
