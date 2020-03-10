#!/user/bin/env python

import numpy as np
import pandas as pd
from helpers import visualise_mask, build_mask
import os
from skimage import exposure
import json
from skimage.io import imread
import random

def chunkify(lst,n):
    return [lst[i::n] for i in range(n)]

def enhance_image(image):
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(64,64))
    # return clahe.apply(image)
    p2, p98 = np.percentile(image, (2, 98))
    return exposure.rescale_intensity(image, in_range=(p2, p98))

# reading in the training set
data = pd.read_csv('./train.csv')

# isolating the file name and class
data['ImageId'], data['ClassId'] = data.ImageId_ClassId.str.split('_', n=1).str
data['ClassId'] = data['ClassId'].astype(np.uint8)

# keep only the images with labels
filtered = data.dropna(subset=['EncodedPixels'], axis='rows')

# squash multiple rows per image into a list
squashed = filtered[['ImageId', 'EncodedPixels', 'ClassId']] \
            .groupby('ImageId', as_index=False) \
            .agg(list)

squashed['npz'] = squashed.ImageId.apply(lambda x: x[:-4] + ".npz")

no_defects = data[['ImageId']].drop_duplicates().set_index('ImageId').drop(squashed.ImageId.tolist()).index.tolist()
class_counts = filtered['ClassId'].value_counts(normalize=True).sort_index()
print("Class counts: %s" % filtered['ClassId'].value_counts().sort_index())
print("No defect instances: %s" % len(no_defects))

outdir = "./out"
os.makedirs(outdir, exist_ok=True)

classes = {}
for clazz in sorted(filtered['ClassId'].unique()):
    members = list(squashed[squashed.ClassId.apply(lambda x: clazz in x)]['npz'])
    random.shuffle(members)
    eval_instances = members[:int(len(members) / 10) + 1]
    train_instances = [m for m in members if m not in eval_instances]
    classes[str(clazz)] = {
        "eval_instances": eval_instances,
        "train_instances": train_instances
    }

for BOOST_FOLD in range(0, 20):
    for NUM_FOLDS in range(2, 21):
        chunked = {}
        for clazz, members in classes.items():
            chunked[str(clazz)] = chunkify(members["train_instances"], NUM_FOLDS)
        folds = []
        for fold_number in range(NUM_FOLDS):
            fold = {}
            for clazz, members in chunked.items():
                ignore_idcs = range(fold_number + 1, fold_number + 1 + BOOST_FOLD)
                ignore_idcs = [ i % NUM_FOLDS for i in ignore_idcs]
                fold[clazz] = {
                    "val": members[fold_number],
                    "ignore": sum([members[i] for i in range(len(members)) if i in ignore_idcs], []),
                    "train": sum([members[i] for i in range(len(members)) if i != fold_number and i not in ignore_idcs], [])
                }
            folds.append(fold)

        with open(os.path.join(outdir, "%s-boost_folds-%s-folds.json" % (BOOST_FOLD, NUM_FOLDS)), "w") as outfile:
            json.dump({
                "folds": folds,
                "class_order": ["1", "2", "3", "4"]
            }, outfile, indent=4)

with open(os.path.join(outdir, "classes.json"), "w") as outfile:
    json.dump({
        "classes": classes,
        "class_order": ["1", "2", "3", "4"]
    }, outfile, indent=4)

for idx, squashed_instance in squashed.iterrows():
    mask = build_mask(squashed_instance['EncodedPixels'], squashed_instance['ClassId']).astype(np.float32)
    name = squashed_instance['ImageId'][:-4]

    if np.max(mask) == 0:
        print("Skipping %s" % name)
        continue
    print(name)

    image = imread(os.path.join("./train", squashed_instance['ImageId']), as_gray=True).astype(np.float32)
    np.savez_compressed(os.path.join(outdir, squashed_instance['ImageId'][:-4]), image=enhance_image(image), mask=mask)
