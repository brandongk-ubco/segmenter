from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)
from augment import DataGeneratorFolder
from matplotlib import pyplot as plt
import numpy as np

def aug_with_crop(image_size=(1600,256), crop_prob=1):
    return Compose([
        RandomCrop(width=image_size[1], height=image_size[0], p=crop_prob),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Transpose(p=0.5),
        ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
        RandomBrightnessContrast(p=0.5),
        RandomGamma(p=0.25),
        IAAEmboss(p=0.25),
        Blur(p=0.01, blur_limit=3),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8)
    ], p=1)


test_generator = DataGeneratorFolder(root_dir='/data/',
                                     image_folder='images/',
                                     mask_folder='masks/',
                                     batch_size=1,
                                     image_size=(256,1600),
                                     image_divisibility=(32,32),
                                     nb_y_features=1,
                                     augmentation=aug_with_crop)

Xtest, ytest = test_generator.__getitem__(0)
print(np.max(Xtest))
print(np.max(ytest))
plt.imshow(Xtest[0])
plt.savefig('/output/x.png')
plt.imshow(ytest[0, :, :, 0])
plt.savefig('/output/y.png')