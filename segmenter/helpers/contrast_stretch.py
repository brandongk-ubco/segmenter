from skimage import exposure
from skimage.util import dtype_limits
import numpy as np


def contrast_stretch(image, min_percentile=2, max_percentile=98):
    p2, p98 = np.percentile(image, (min_percentile, max_percentile))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    min_val, max_val = dtype_limits(image, clip_negative=True)
    image = np.clip(image, min_val, max_val)
    return image
