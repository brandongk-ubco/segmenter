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

    def postprocess(self, img, mask):
        if self.job_config["RECENTER"]:
            img = img - np.mean(img)
        return img.astype('float16'), mask.astype('float16')

    def augment(self, img, mask):
        if self.augmentations(self.job_config, img.shape) is None:
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
    from augmentations import train_augments, val_augments
    from matplotlib import pyplot as plt
    from config import get_config
    from helpers import generate_for_augments
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

    job_config = get_config()
    clazz = os.environ.get("CLASS", job_config["CLASSES"][0])
    fold = int(os.environ.get("FOLD", 0))
    path = os.environ.get("DATA_PATH")

    val_generators, val_dataset, num_val_images = generate_for_augments(clazz, fold, val_augments, job_config, path=path, mode="val", repeat=True)
    train_generators, train_dataset, num_train_images = generate_for_augments(clazz, fold, val_augments, job_config, path=path, mode="train", shuffle=True, repeat=True)

    seen_imgs = []
    for i, (img, mask) in enumerate(train_dataset.as_numpy_iterator()):
        print(i)
        if i > num_train_images:
            break
        img = img[:,:,0].astype('float32')
        mask = mask[:,:,0].astype('float32')

        for seen in seen_imgs:
            if (seen==img).all():
                print("We've seen this before!")
            else:
                seen_imgs.append(img)

        plt.subplot(2,1,1)
        plt.imshow(img, cmap='gray')
        plt.subplot(2,1,2)
        plt.imshow(mask, cmap='gray')
        plt.show()
