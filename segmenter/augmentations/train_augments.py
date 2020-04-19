from albumentations import Compose, Normalize, HorizontalFlip, RandomResizedCrop, ElasticTransform, RandomResizedCrop, VerticalFlip, RandomGamma
from cv2 import INTER_LINEAR


def train_augments(augments_config, mean, std, image_size):
    box_scale = min(image_size)
    return Compose([
        HorizontalFlip(p=augments_config["HORIZONTAL_FLIP_PR"]),
        VerticalFlip(p=augments_config["VERTICAL_FLIP_PR"]),
        RandomResizedCrop(image_size[0],
                          image_size[1],
                          scale=(1, 1 + augments_config["RESCALE_PERCENTAGE"]),
                          ratio=(image_size[1] / image_size[0],
                                 image_size[1] / image_size[0]),
                          interpolation=INTER_LINEAR,
                          p=augments_config["RESCALE_PR"]),
        ElasticTransform(alpha=box_scale,
                         sigma=box_scale * 0.05,
                         alpha_affine=box_scale * 0.03,
                         p=augments_config["ELASTIC_TRANSFORM_PR"]),
        RandomGamma(gamma_limit=(80, 120), p=augments_config["GAMMA_PR"]),
        Normalize(max_pixel_value=1.0,
                  mean=mean,
                  std=std,
                  p=augments_config["NORMALIZE_PR"])
    ],
                   p=1)
