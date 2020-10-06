import json
from segmenter.helpers.p_tqdm import p_map as mapper
import glob
import numpy as np
import os
from skimage import measure
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
import cv2


class DatasetVisualizer:
    def __init__(self, src_dir):
        self.src_dir = src_dir
        with open(os.path.join(self.src_dir, "classes.json"),
                  "r") as json_file:
            data = json.load(json_file)
        self.classes = data["class_order"]
        self.color_map = [
            c for i, c in enumerate(cm.get_cmap("tab10").colors)
            if i in [1, 2, 3, 6]
        ]

    def execute_result(self, result_path):
        name = os.path.basename(result_path)[:-4]

        outfile = os.path.join(os.path.dirname(result_path),
                               "{}.png".format(name))

        instance_data = np.load(result_path)
        image = instance_data["image"]
        mask = instance_data["mask"]

        alphas = np.ones_like(image)

        highlighted_image = np.dstack(
            (image.copy() * alphas, image.copy() * alphas,
             image.copy() * alphas))

        for i in range(len(self.classes)):
            mask_i = mask[:, :, i].copy().astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask_i, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            contour_mask = np.zeros_like(mask_i)
            cv2.drawContours(contour_mask, contours, -1, 1,
                             int(np.mean(mask_i.shape) * .007))
            alphas[contour_mask == 1] = 0
            highlighted_image[contour_mask == 1, :] = self.color_map[i]

        legend = [
            Line2D(
                [0],
                [0],
                color=l,
                lw=4,
            ) for l in self.color_map[0:len(self.classes)]
        ]

        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray', interpolation='none')
        ax.imshow(highlighted_image, alpha=1, interpolation='none')
        ax.axis('off')
        if "kits19" in self.src_dir:
            fig.suptitle(name, y=1, fontsize=16)
        elif "severstal" in self.src_dir:
            fig.suptitle(name, y=0.7, fontsize=16)
        else:
            raise ValueError("Couldn't determine dataset.")
        plt.legend(legend,
                   self.classes,
                   bbox_to_anchor=(0, 1, 1, 0.2),
                   loc="lower left",
                   frameon=False,
                   ncol=len(self.classes))

        plt.savefig(outfile, dpi=150, bbox_inches='tight')

        plt.close()

    def execute(self):
        mapper(self.execute_result, self.collect_results())

    def collect_results(self):
        return sorted(glob.glob("{}/*.npz".format(self.src_dir)))
