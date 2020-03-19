from segmentation_models.losses import DiceLoss, BinaryFocalLoss, binary_crossentropy
from .BinaryCrossentropyWithLogits import BinaryCrossentropyWithLogits

def get_loss(loss_config):
    loss = loss_config["DICE_MULTIPLIER"] * DiceLoss(beta=loss_config["DICE_BETA"])
    loss += loss_config["BFL_MULTIPLIER"] * BinaryFocalLoss(gamma=loss_config["BFL_GAMMA"])
    loss += loss_config["BCE_MULTIPLIER"] * binary_crossentropy
    loss += loss_config["BCE_LOGITS_MULTIPLIER"] * BinaryCrossentropyWithLogits()
    return loss