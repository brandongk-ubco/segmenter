from segmentation_models.base import Metric, functional as F

def specificity(gt, pr, class_weights=1, class_indexes=None, per_image=False, threshold=None, **kwargs):
    backend = kwargs['backend']

    gt, pr = F.gather_channels(gt, pr, indexes=class_indexes, **kwargs)
    pr = F.round_if_needed(pr, threshold, **kwargs)
    axes = F.get_reduce_axes(per_image, **kwargs)

    tp = backend.sum(gt * pr, axis=axes)
    fp = backend.sum(pr, axis=axes) - tp
    n = backend.cast(backend.prod(backend.shape(pr)), fp.dtype)
    
    score = 1 - fp / n

    return score

class Specificity(Metric):
    def __init__(
            self,
            class_weights=None,
            class_indexes=None,
            threshold=None,
            per_image=False,
            name=None,
    ):
        name = name or 'specificity'
        super().__init__(name=name)
        self.class_weights = class_weights if class_weights is not None else 1
        self.class_indexes = class_indexes
        self.threshold = threshold
        self.per_image = per_image

    def __call__(self, gt, pr):
        return specificity(
            gt,
            pr,
            class_weights=self.class_weights,
            class_indexes=self.class_indexes,
            per_image=self.per_image,
            threshold=self.threshold,
            **self.submodules
        )