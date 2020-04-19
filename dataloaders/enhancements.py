from skimage import exposure
import numpy as np

def contrast_stretch(image, min_percentile=2, max_percentile=98):
    p2, p98 = np.percentile(image, (min_percentile, max_percentile))
    return exposure.rescale_intensity(image, in_range=(p2, p98))