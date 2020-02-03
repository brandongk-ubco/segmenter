from albumentations import *
import cv2

def augment(image_size):
    box_scale = min(image_size)
    rescale_percentage = 0.1
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        OneOf([
            Compose([
                RandomScale(scale_limit=(0, rescale_percentage), p=1, interpolation=cv2.INTER_CUBIC),
                RandomCrop(image_size[0], image_size[1], p=1),
            ], p=0.5),
            RandomResizedCrop(
                image_size[0],
                image_size[1],
                scale=(1 - rescale_percentage, 1 + rescale_percentage),
                ratio=(image_size[1]/image_size[0], image_size[0]/image_size[1]),
                interpolation=cv2.INTER_CUBIC,
                p=0.5
            )
        ], p=1),
        # ElasticTransform(p=0.5, alpha=box_scale, sigma=box_scale * 0.05, alpha_affine=box_scale * 0.03),
    ], p=1)
