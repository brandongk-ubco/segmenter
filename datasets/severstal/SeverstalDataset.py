import cv2 as cv
import numpy as np
import random
import pandas as pd
from skimage.io import imread
import os
import sys
sys.path.append("..")

from BaseDataset import BaseDataset
from enhancements import contrast_stretch

current_dir = os.path.dirname(os.path.realpath(__file__))

def rle_to_mask(lre, shape=(1600,256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return 
    
    returns: numpy array with dimensions of shape parameter
    '''    
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])
    
    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1
    
    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]
    
    # build the mask
    h, w = shape
    mask = np.zeros(h*w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1
    
    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)

def build_mask(encodings, labels):
    """ takes a pair of lists of encodings and labels, 
        and turns them into a 3d numpy array of shape (256, 1600, 4) 
    """
    
    # initialise an empty numpy array 
    mask = np.zeros((256,1600,4), dtype=np.uint8)
   
    # building the masks
    for rle, label in zip(encodings, labels):
        # classes are [1, 2, 3, 4], corresponding indeces are [0, 1, 2, 3]
        index = label - 1
        
        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for 
        # numpy and openCV handling width and height in reverse order 
        mask[:,:,index] = rle_to_mask(rle).T
    
    return mask

class SeverstalDataset(BaseDataset):

    def __init__(self, path=current_dir):
        self.path = os.path.abspath(path)
        # reading in the training set
        data = pd.read_csv(os.path.join(self.path, 'train.csv'))

        # isolating the file name and class
        data['ImageId'], data['ClassId'] = data.ImageId_ClassId.str.split('_', n=1).str
        data['ClassId'] = data['ClassId'].astype(np.uint8)

        # keep only the images with labels
        self.filtered = data.dropna(subset=['EncodedPixels'], axis='rows')

        # squash multiple rows per image into a list
        self.squashed = self.filtered[['ImageId', 'EncodedPixels', 'ClassId']] \
                    .groupby('ImageId', as_index=False) \
                    .agg(list)

        self.no_defects = data[['ImageId']].drop_duplicates().set_index('ImageId').drop(self.squashed.ImageId.tolist()).index.tolist()
        print("No defect instances: %s" % len(self.no_defects))

    def get_classes(self):
        return sorted([str(c) for c in self.filtered['ClassId'].sort_index().unique().tolist()])

    def get_class_members(self):
        classes = {}
        for clazz in self.get_classes():
            members = self.filtered[self.filtered["ClassId"] == clazz]["ImageId"].tolist()
            members = [os.path.splitext(m)[0] for m in members]
            random.shuffle(members)
            eval_instances = members[:int(len(members) / 10) + 1]
            train_instances = [m for m in members if m not in eval_instances]
            classes[str(clazz)] = {
                "eval_instances": eval_instances,
                "train_instances": train_instances
            }
        return classes

    def iter_instances(self):
        for idx, instance in self.squashed.iterrows():
            yield instance

    def get_name(self, instance):
        return instance['ImageId'][:-4]

    def get_image(self, instance):
        return imread(os.path.join(self.path, "train", instance['ImageId']), as_gray=True).astype(np.float32)

    def get_masks(self, instance):
        return build_mask(instance['EncodedPixels'], instance['ClassId']).astype(np.float32)

    def enhance_image(self, image):
        return contrast_stretch(image)
