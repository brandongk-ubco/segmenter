from albumentations import *
import cv2

def train_augment(job_config, image_size, hflip=0.5, vflip=0.5):
    box_scale = min(image_size)
    return Compose([
        HorizontalFlip(p=hflip),
        VerticalFlip(p=vflip),
        RandomResizedCrop(
            image_size[0], image_size[1],
            scale=(1, 1 + job_config["RESCALE_PERCENTAGE"]),
            ratio=(image_size[1]/image_size[0], image_size[1]/image_size[0]),
            interpolation=cv2.INTER_LINEAR,
            p=job_config["RESCALE_PR"]
        ),
        ElasticTransform(
            alpha=box_scale,
            sigma=box_scale * 0.05,
            alpha_affine=box_scale * 0.03,
            p=job_config["ELASTIC_TRANSFORM_PR"]
        )
    ], p=1)

def train_augments():
    return [
        lambda job_config, image_size: train_augment(job_config, image_size)
    ]

def val_augments():
    return [
        None,
        lambda job_config, image_size : HorizontalFlip(p=1),
        lambda job_config, image_size : VerticalFlip(p=1),
        lambda job_config, image_size : Compose([
            HorizontalFlip(p=1),
            VerticalFlip(p=1),
        ], p=1)
    ]

def predict_augments():
    return [
        None
    ]