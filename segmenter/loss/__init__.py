from segmentation_models.losses import DiceLoss, BinaryFocalLoss
from .BinaryCrossentropy import BinaryCrossentropy


def get_loss(loss_config):
    loss = loss_config["DICE_MULTIPLIER"] * DiceLoss(
        beta=loss_config["DICE_BETA"])
    loss += loss_config["BFL_MULTIPLIER"] * BinaryFocalLoss(
        gamma=loss_config["BFL_GAMMA"])
    loss += loss_config["BCE_MULTIPLIER"] * BinaryCrossentropy(
        loss_config["BCE_FROM_LOGITS"])
    return loss