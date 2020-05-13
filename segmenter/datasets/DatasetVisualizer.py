import json
from segmenter.helpers.p_tqdm import p_map as mapper
import glob
import numpy as np
import os
from skimage import measure
from matplotlib import pyplot as plt


class DatasetVisualizer:
    def __init__(self, dataset, src_dir):
        self.dataset = dataset
        self.src_dir = src_dir
        self.classes = self.dataset.get_classes()

    def execute_result(self, result_path):
        name = os.path.basename(result_path)[:-4]

        outfile = os.path.join(os.path.dirname(result_path),
                               "{}.jpg".format(name))
        if os.path.exists(outfile):
            return

        instance_data = np.load(result_path)
        image = instance_data["image"]
        mask = instance_data["mask"]

        fig = plt.figure(constrained_layout=True)

        gs = fig.add_gridspec(2, mask.shape[-1])
        fig_image = fig.add_subplot(gs[:, 0])
        fig_image.axis('off')
        fig_image.set_title(name)
        fig_image.imshow(image, cmap='gray')

        for mask_idx in range(mask.shape[-1]):
            fig_mask = fig.add_subplot(gs[mask_idx, 1])
            fig_mask.axis('off')
            fig_mask.set_title(self.classes[mask_idx])
            fig_mask.imshow(mask[:, :, mask_idx], cmap='gray')

        plt.savefig(outfile,
                    dpi=70,
                    bbox_inches='tight',
                    pad_inches=0.5,
                    quality=90,
                    optimize=True)

        plt.close()

    def execute(self):
        mapper(self.execute_result, sorted(self.collect_results()))

    def collect_results(self):
        return glob.glob("{}/*.npz".format(self.src_dir))
