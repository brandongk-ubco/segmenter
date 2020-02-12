from albumentations import *

def train_augment(job_config, image_size):
    box_scale = min(image_size)
    rescale_percentage = 0.1
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        OneOf([
            Compose([
                RandomScale(scale_limit=(0, rescale_percentage), p=1),
                RandomCrop(image_size[0], image_size[1], p=1),
            ], p=0.5),
            RandomResizedCrop(
                image_size[0],
                image_size[1],
                scale=(1 - rescale_percentage, 1 + rescale_percentage),
                ratio=(image_size[1]/image_size[0], image_size[0]/image_size[1]),
                p=0.5
            )
        ], p=1),
        # ElasticTransform(p=job_config["ELASTIC_TRANSFORM_PR"], alpha=box_scale, sigma=box_scale * 0.05, alpha_affine=box_scale * 0.03),
    ], p=1)

def val_augment(job_config, image_size):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
    ], p=1)
