from segmentation_models.base import Metric, functional as F
from .Specificity import specificity


class FallOut(Metric):
    def __init__(
        self,
        class_weights=None,
        class_indexes=None,
        threshold=None,
        per_image=False,
        name=None,
    ):
        name = name or 'fallout'
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image

    def __call__(self, gt, pr):
        return 1 - specificity(gt,
                               pr,
                               class_weights=self.class_weights,
                               class_indexes=self.class_indexes,
                               per_image=self.per_image,
                               threshold=self.threshold,
                               **self.submodules)
