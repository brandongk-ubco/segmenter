import os
import tensorflow as tf
from keras.utils import Sequence
from skimage.io import imread
from skimage import img_as_float32, img_as_ubyte
from sklearn.utils import shuffle
import albumentations as albu
from albumentations import Resize
import numpy as np

class DataGeneratorFolder(Sequence):
    def __init__(self, root_dir=r'../data/', image_folder='images/', mask_folder='masks/',
                 batch_size=4, image_size=(256, 1024), nb_y_features=1,
                 augmentation=None,
                 suffle=True,
                 image_divisibility=32):
        self.image_filenames = [os.path.abspath(os.path.join(root_dir, image_folder, i)) for i in os.listdir(os.path.join(root_dir, image_folder))]
        self.mask_names = [os.path.abspath(os.path.join(root_dir, mask_folder, i)) for i in os.listdir(os.path.join(root_dir, mask_folder))]
        self.batch_size = batch_size
        self.currentIndex = 0
        self.augmentation = augmentation
        self.image_size = image_size
        self.image_divisibility = image_divisibility
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.suffle = suffle

    def __len__(self):
        """
        Calculates size of batch
        """
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.suffle == True:
            self.image_filenames, self.mask_names = shuffle(self.image_filenames, self.mask_names)

    def read_image_mask(self, image_name, mask_name):
        image = imread(image_name)
        mask = imread(mask_name, as_gray=True)
        return img_as_float32(image), img_as_ubyte(mask)

    def __getitem__(self, index):
        """
        Generate one batch of data
        
        """
        # Generate indexes of the batch
        data_index_min = int(index * self.batch_size)
        data_index_max = int(min((index + 1) * self.batch_size, len(self.image_filenames)))

        indexes = self.image_filenames[data_index_min:data_index_max]

        this_batch_size = len(indexes)  # The last batch can be smaller than the others

        # Defining dataset
        X = np.empty((this_batch_size, self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size[0], self.image_size[1], self.nb_y_features), dtype=np.uint8)

        for i, sample_index in enumerate(indexes):

            X_sample, y_sample = self.read_image_mask(self.image_filenames[index * self.batch_size + i],
                                                      self.mask_names[index * self.batch_size + i])

            # if augmentation is defined, we assume its a train set
            if self.augmentation is not None:
                # Augmentation code
                augmented = self.augmentation(self.image_size)(image=X_sample, mask=y_sample)
                image_augm = augmented['image'].reshape(self.image_size[0], self.image_size[1], augmented['image'].shape[2])
                mask_augm = augmented['mask'].reshape(self.image_size[0], self.image_size[1], self.nb_y_features)
                X[i, ...] = np.clip(image_augm, a_min=0., a_max=1.)
                y[i, ...] = mask_augm

            # if augmentation isnt defined, we assume its a test set. 
            # Because test images can have different sizes we resize it to be divisable by 32
            elif self.augmentation is None and self.batch_size == 1:

                augmented = Resize(
                    height=(X_sample.shape[0] // self.image_divisibility[0]) * self.image_divisibility[0],
                    width=(X_sample.shape[1] // self.image_divisibility[1]) * self.image_divisibility[1]
                )(image=X_sample, mask=y_sample)

                X_sample, y_sample = augmented['image'], augmented['mask']

                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], augmented['image'].shape[2]), y_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features)

        return X, y
