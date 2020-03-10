from segmentation_models.losses import DiceLoss, BinaryFocalLoss, binary_crossentropy

def get_loss(loss_config):
    loss = loss_config["DICE_MULTIPLIER"] * DiceLoss(beta=loss_config["BETA"])
    loss += loss_config["BFL_MULTIPLIER"] * BinaryFocalLoss(gamma=loss_config["GAMMA"])
    loss += loss_config["BCE_MULTIPLIER"] * binary_crossentropy
    return loss