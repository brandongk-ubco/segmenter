from segmentation_models.losses import DiceLoss, binary_crossentropy

def get_loss(loss):
    if loss["NAME"] == "dice_bce":
        return DiceLoss(beta=loss["BETA"]) + binary_crossentropy
    if loss["NAME"] == "dice":
        return DiceLoss(beta=loss["BETA"])
    if loss["NAME"] == "bce":
        return binary_crossentropy
    raise ValueError("Loss %s not defined" % loss)