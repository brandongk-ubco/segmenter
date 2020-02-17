import os
import json
import random
import numpy as np
import io
from glob import glob

class CachedFilereader:

    cache = {}

    def read(self, filename):
        if filename not in self.cache:
            with open(filename, "rb") as newfile:
                self.cache[filename] = newfile.read()

        return io.BytesIO(self.cache[filename])

cache = CachedFilereader()

class DataGenerator:

    def __init__(self, clazz, fold, mode="train", path="/data", augmentations=None, job_config=None):
        with open(os.path.join(path, "%s-trainbehind-%s-folds.json" % (job_config["TRAIN_BEHIND"], job_config["FOLDS"])), "r") as json_file:
            data = json.load(json_file)

        self.augmentations = augmentations
        self.path = path
        if fold is None and mode in ["predict", "evaluate"]:
            self.data = [f for f in  glob(os.path.join(path, '*')) if os.path.isfile(f) and f.endswith(".npz")]
        elif fold is not None and mode in ["train", "val"]:
            self.data = sorted(data["folds"][fold][clazz][mode])
        else:
            raise ValueError("Invalid combination: mode %s / fold %s" % (mode, fold))
        self.mask_index = data["class_order"].index(clazz)
        self.job_config = job_config

    def size(self):
        return len(self.data)

    def recenter(self, img, mask):
        img = img - np.mean(img)
        return img.astype('float16'), mask.astype('float16')

    def augment(self, img, mask):
        if self.augmentations is None:
            return img, mask
        mask_coverage_before = np.sum(mask)
        mask_coverage_after = 0
        while mask_coverage_after / mask_coverage_before < 0.5:
            augmented = self.augmentations(self.job_config, img.shape)(image=img, mask=mask)
            mask_coverage_after = np.sum(augmented['mask'])

        return augmented["image"], augmented["mask"]

    # This works, but we can't stack images of different dimensions without padding...
    # So, this will only work in practice for a batch size of 1....
    def crop(self, img, mask):
        keep = img > 0.1
        keep_idx = np.ix_(keep.any(1)[:,0], keep.any(0)[:,0])
        return img[keep_idx], mask[keep_idx]

    def generate(self):
        for i in range(len(self.data)):
            filename = os.path.join(self.path, self.data[i])
            i_data = np.load(cache.read(os.path.join(self.path, self.data[i])))
            img = i_data["image"]
            mask = i_data["mask"][:, :, self.mask_index]
            if len(img.shape) == 2:
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            yield img, mask

if __name__ == "__main__":
    from augmentations import train_augment, val_augments
    from matplotlib import pyplot as plt
    from config import get_config

    for val_augment in val_augments():

        generator = DataGenerator("1", 0, mode="val", path=os.environ.get("DATA_PATH"), augmentations=val_augment, job_config=get_config())
        img, mask = next(generator.generate())
        img = img[:,:,0]
        mask = mask[:,:,0]

        augmented, mask = generator.augment(img, mask)
        plt.subplot(2,1,1)
        plt.imshow(img, cmap='gray')
        plt.subplot(2,1,2)
        plt.imshow(augmented, cmap='gray')
        plt.show()
