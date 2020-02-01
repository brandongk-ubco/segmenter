from tensorflow.keras import backend
from segmentation_models.metrics import f1_score
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy, DiceLoss, BinaryFocalLoss
from segmentation_models.base import Loss

class NormalizedFocalLoss(Loss):

    def __call__(self, gt, pr, **kwargs):
        dl = dice_loss(gt, pr, **kwargs)
        bfl = BinaryFocalLoss(alpha=1, gamma=5)(gt, pr, **kwargs)
        bce = binary_crossentropy(gt, pr, **kwargs)

        ce_loss = bfl + bce
        
        return ce_loss + ce_loss * dl / backend.mean(ce_loss)