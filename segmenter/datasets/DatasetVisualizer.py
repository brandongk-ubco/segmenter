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
                               "{}.png".format(name))
        # if os.path.exists(outfile):
        #     return

        instance_data = np.load(result_path)
        image = instance_data["image"]
        mask = instance_data["mask"]

        fig, axs = plt.subplots(mask.shape[-1] + 1, 1)
        axs[0].axis('off')
        axs[0].set_title(name)
        axs[0].imshow(image, cmap='gray')

        for mask_idx in range(mask.shape[-1]):
            ax = axs[mask_idx + 1]
            ax.set_frame_on(False)
            ax.imshow(mask[:, :, mask_idx], cmap='gray')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_ylabel(self.classes[mask_idx])

        plt.savefig(outfile, dpi=70, bbox_inches='tight', pad_inches=0.5)

        plt.close()

    def execute(self):
        mapper(self.execute_result, self.collect_results())

    def collect_results(self):
        return sorted(glob.glob("{}/*.npz".format(self.src_dir)))
