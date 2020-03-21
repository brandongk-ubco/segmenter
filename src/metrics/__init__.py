import sys
sys.path.append("..")

from .Specificity import Specificity
from .FallOut import FallOut

from segmentation_models.metrics import FScore, Precision, Recall, IOUScore
from segmentation_models.losses import DiceLoss, BinaryFocalLoss, binary_crossentropy, binary_focal_loss, jaccard_loss
from loss import BinaryCrossentropy

def get_metrics(threshold, loss):

    metrics = {
        "f1-score": FScore(threshold=threshold),
        "iou_score": IOUScore(threshold=threshold),
        "precision": Precision(threshold=threshold),
        "recall": Recall(threshold=threshold),
        "specificity": Specificity(threshold=threshold),
        "dice_loss": DiceLoss(beta=1),
        "binary_crossentropy_loss": BinaryCrossentropy(from_logits=loss["BCE_FROM_LOGITS"]),
        "binary_focal_loss": binary_focal_loss,
        "jaccard_loss": jaccard_loss
    }

    if loss["DICE_MULTIPLIER"] > 0 and loss["DICE_BETA"] != 1.0:
        beta = loss["BETA"]
        dl_beta_loss = DiceLoss(beta=beta)
        dl_beta_loss.name = 'd{}-loss'.format(beta)
        metrics[dl_beta_loss.name] = dl_beta_loss

        f_beta_score = FScore(beta=beta, threshold=threshold)
        metrics[f_beta_score.name] = f_beta_score

    if loss["BFL_MULTIPLIER"] > 0 and loss["BFL_GAMMA"] != 1.0:
        gamma = loss["BFL_GAMMA"]
        bfl_gamma_loss = BinaryFocalLoss(gamma=gamma)
        bfl_gamma_loss.name = 'bfl{}-loss'.format(gamma)
        metrics[bfl_gamma_loss.name] = bfl_gamma_loss

    return metrics