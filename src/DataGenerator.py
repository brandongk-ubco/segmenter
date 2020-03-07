import os
import json
import random
import numpy as np
import io
import glob
from tensorflow.keras import backend as K

class CachedFilereader:

    cache = {}

    def read(self, filename):
        if filename not in self.cache:
            with open(filename, "rb") as newfile:
                self.cache[filename] = newfile.read()

        return io.BytesIO(self.cache[filename])

cache = CachedFilereader()

class DataGenerator:

    def __init__(self, clazz, fold, shuffle=False, method="include", mode="train", path="/data", augmentations=None, job_config=None):

        self.augmentations = augmentations
        self.path = path
        if fold is None and mode == "evaluate":
            with open(os.path.join(path, "classes.json"), "r") as json_file:
                self.data = json.load(json_file)
                self.image_files = sorted(self.data["classes"][clazz]["eval_instances"])
        elif fold is not None and mode in ["train", "val"]:
            with open(os.path.join(path, "%s-boost_folds-%s-folds.json" % (job_config["BOOST_FOLDS"], job_config["FOLDS"])), "r") as json_file:
                self.data = json.load(json_file)
                self.image_files = sorted(self.data["folds"][fold][clazz][mode])
        else:
            raise ValueError("Invalid combination: mode %s / fold %s" % (mode, fold))
            
        if method == "exclude":
            all_files = [ f for f in os.listdir(path) if f.endswith(".npz") ]
            self.image_files = list(filter(lambda f: f not in self.image_files, all_files))
            
        self.image_files = list(map(lambda f: os.path.abspath(os.path.join(self.path, f)), self.image_files))

        if shuffle:
            random.shuffle(self.image_files)

        self.mask_index = self.data["class_order"].index(clazz)
        self.job_config = job_config

    def size(self):
        return len(self.image_files)

    def postprocess(self, img, mask):
        if self.job_config["RECENTER"]:
            img = img - np.mean(img)
        return img.astype(K.floatx()), mask.astype(K.floatx())

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

    def _generate(self, idx):
        filename = os.path.join(self.path, self.image_files[idx])
        i_data = np.load(cache.read(filename))
        img = i_data["image"]
        mask = i_data["mask"][:, :, self.mask_index]
        if len(img.shape) == 2:
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

        return img, mask

    def generate(self):
        for i in range(len(self.image_files)):
            yield self._generate(i)

if __name__ == "__main__":
    from augmentations import train_augments, val_augments
    from matplotlib import pyplot as plt
    from config import get_config
    from helpers import generate_for_augments
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import tensorflow as tf

    job_config = get_config()
    K.set_floatx(job_config["PRECISION"])
    clazz = os.environ.get("CLASS", job_config["CLASSES"][0])
    fold = int(os.environ.get("FOLD", 0))
    path = os.environ.get("DATA_PATH")

    val_generators, augmented_val_dataset, num_val_images = generate_for_augments(clazz, fold, val_augments, job_config, path=path, mode="val", repeat=True)
    train_generators, augmented_train_dataset, num_train_images = generate_for_augments(clazz, fold, train_augments, job_config, path=path, mode="train", shuffle=True, repeat=True)

    seen_imgs = []
    repeated = 0
    for i, (augmented_img, augmented_mask) in enumerate(augmented_train_dataset.as_numpy_iterator()):
        if i >= 2*num_train_images:
            break

        print(mask.dtype)
        augmented_img = augmented_img[:,:,0]
        augmented_mask = augmented_mask[:,:,0]

        been_seen = False
        for seen in seen_imgs:
            if (seen==augmented_img).all():
                been_seen = True
                break

        if not been_seen:
            seen_imgs.append(augmented_img)

        if not been_seen and i > num_train_images:
            print("New image in 2nd iteration!")

        if been_seen and i < num_train_images:
            print("Repeated image in 1st iteration!")

        if been_seen and i > num_train_images:
            repeated += 1
            print("Repeated image in 2nd iteration!")

        # plt.subplot(2,1,1)
        # plt.imshow(augmented_img.astype('float32'), cmap='gray')
        # plt.subplot(2,1,2)
        # plt.imshow(augmented_mask.astype('float32'), cmap='gray')
        # plt.show()

    # This should be roughly 25% for 50% rotation augments
    # This should be 100% for val augments
    print("%s/%s" % (repeated, num_train_images))
