from .Specificity import Specificity
from .FallOut import FallOut

from segmentation_models.metrics import FScore, Precision, Recall, IOUScore
from segmentation_models.losses import DiceLoss, binary_crossentropy, binary_focal_loss, jaccard_loss

def get_metrics(threshold, loss):

    metrics = {
        "f1-score": FScore(threshold=threshold),
        "iou_score": IOUScore(threshold=threshold),
        "precision": Precision(threshold=threshold),
        "recall": Recall(threshold=threshold),
        "specificity": Specificity(threshold=threshold),
        "dice_loss": DiceLoss(beta=1),
        "binary_crossentropy": binary_crossentropy,
        "binary_focal_loss": binary_focal_loss,
        "jaccard_loss": jaccard_loss
    }

    if "BETA" in loss and loss["BETA"] != 1:
        beta = loss["BETA"]
        dl_beta_loss = DiceLoss(beta=beta)
        dl_beta_loss.name = 'd{}-loss'.format(beta)
        metrics[dl_beta_loss.name] = dl_beta_loss

        f_beta_score = FScore(beta=beta, threshold=threshold)
        metrics[f_beta_score.name] = f_beta_score

    return metrics