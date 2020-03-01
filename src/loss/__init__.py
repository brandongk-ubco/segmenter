from segmentation_models.losses import dice_loss, binary_crossentropy

def get_loss(loss):
    if loss == "combined":
        return dice_loss + binary_crossentropy
    if loss == "dice":
        return dice_loss
    if loss == "bce":
        return binary_crossentropy
    raise ValueError("Loss %s not defined" % loss)