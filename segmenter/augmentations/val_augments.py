from albumentations import *
import cv2


def val_augments(augments_config, mean, std, image_size):
    return Compose([
        Normalize(max_pixel_value=1.0, mean=mean, std=std,
                  p=augments_config["NORMALIZE_PR"])
    ], p=1)
