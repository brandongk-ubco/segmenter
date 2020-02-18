from tensorflow.keras import backend as K
from segmentation_models.metrics import f1_score
from segmentation_models.losses import binary_focal_loss, dice_loss, binary_crossentropy, DiceLoss, BinaryFocalLoss
from segmentation_models.base import Loss
from segmentation_models.metrics import FScore
import tensorflow as tf

class NormalizedFocalLoss(Loss):

    def __init__(self, threshold, alpha=1, gamma=5, **kwargs):
        super(NormalizedFocalLoss, self).__init__(**kwargs)
        self.threshold = threshold
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, gt, pr, **kwargs):
        dl = dice_loss(gt, pr, **kwargs)

        f_score = FScore(threshold=self.threshold)(gt, pr, **kwargs)

        bfl = BinaryFocalLoss(alpha=self.alpha, gamma=self.gamma)(gt, pr, **kwargs)
        bce = binary_crossentropy(gt, pr, **kwargs)

        ce_loss = bfl + bce
        return ce_loss + ce_loss * (dl + 1 - f_score) / (2 * K.mean(ce_loss))