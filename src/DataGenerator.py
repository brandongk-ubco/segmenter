import os
import json
import random
import numpy as np

class DataGenerator:

    def __init__(self, clazz, fold, mode="train", path="../data", augmentations=None, job_config=None):
        with open(os.path.join(path, "folds.json"), "r") as json_file:
            data = json.load(json_file)

        self.augmentations = augmentations
        self.path = path
        self.fold_data = sorted(data["folds"][fold][clazz][mode])
        self.shuffle()
        self.mask_index = data["class_order"].index(clazz)
        self.job_config = job_config

    def shuffle(self):
        random.shuffle(self.fold_data)

    def size(self):
        return len(self.fold_data)

    def generate(self):
        while True:
            for i in range(len(self.fold_data)):
                filename = os.path.join(self.path, self.fold_data[i])
                i_data = np.load(filename)
                img = i_data["image"]
                mask = i_data["mask"][:, :, self.mask_index]
                if len(img.shape) == 2:
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

                if self.augmentations is not None:
                    mask_coverage_before = np.sum(mask)
                    mask_coverage_after = 0
                    while mask_coverage_after / mask_coverage_before < 0.5:
                        augmented = self.augmentations(self.job_config, img.shape)(image=img, mask=mask)
                        mask_coverage_after = np.sum(augmented['mask'])

                yield img, mask

if __name__ == "__main__":
    from augmentations import augment
    from matplotlib import pyplot as plt
    generator = DataGenerator("1", 0, mode="train", augmentations=augment)
    for img, mask in generator.generate():
        plt.subplot(2,1,1)
        plt.imshow(img[:,:,0], cmap='gray')
        plt.subplot(2,1,2)
        plt.imshow(mask[:,:,0], cmap='gray')
        plt.show()
