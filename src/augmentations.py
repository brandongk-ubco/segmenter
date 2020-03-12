from albumentations import *
import cv2

def train_augments(augments_config, image_size):
    box_scale = min(image_size)
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomResizedCrop(
            image_size[0], image_size[1],
            scale=(1, 1 + augments_config["RESCALE_PERCENTAGE"]),
            ratio=(image_size[1]/image_size[0], image_size[1]/image_size[0]),
            interpolation=cv2.INTER_LINEAR,
            p=augments_config["RESCALE_PR"]
        ),
        ElasticTransform(
            alpha=box_scale,
            sigma=box_scale * 0.05,
            alpha_affine=box_scale * 0.03,
            p=augments_config["ELASTIC_TRANSFORM_PR"]
        )
    ], p=1)

def val_augments(augments_config, image_size):
    return None

def predict_augments(augments_config, image_size):
    return None