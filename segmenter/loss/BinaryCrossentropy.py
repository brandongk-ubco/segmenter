from segmentation_models.base import Loss

def binary_crossentropy(gt, pr, from_logits, **kwargs):
    backend = kwargs['backend']
    return backend.mean(backend.binary_crossentropy(gt, pr, from_logits=from_logits))

class BinaryCrossentropy(Loss):
    """Creates a criterion that measures the Binary Cross Entropy between the
    ground truth (gt) and the prediction (pr).
    .. math:: L(gt, pr) = - gt \cdot \log(pr) - (1 - gt) \cdot \log(1 - pr)
    Returns:
        A callable ``binary_crossentropy`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    Example:
    .. code:: python
        loss = BinaryCrossentropy()
        model.compile('SGD', loss=loss)
    """

    def __init__(self, from_logits=False):
        super().__init__(name='binary_crossentropy_loss')
        self.from_logits=from_logits

    def __call__(self, gt, pr):
        return binary_crossentropy(gt, pr, from_logits=self.from_logits, **self.submodules)