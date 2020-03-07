from .Specificity import Specificity
from .FallOut import FallOut

from segmentation_models.metrics import FScore, Precision, Recall, IOUScore
from segmentation_models.losses import dice_loss, binary_crossentropy, binary_focal_loss, jaccard_loss

def get_metrics(threshold):
    return {
        "f1-score": FScore(threshold=threshold),
        "f2-score": FScore(beta=2, threshold=threshold),
        "iou_score": IOUScore(threshold=threshold),
        "precision": Precision(threshold=threshold),
        "recall": Recall(threshold=threshold),
        "specificity": Specificity(threshold=threshold),
        "dice_loss": dice_loss,
        "binary_crossentropy": binary_crossentropy,
        "binary_focal_loss": binary_focal_loss,
        "jaccard_loss": jaccard_loss
    }