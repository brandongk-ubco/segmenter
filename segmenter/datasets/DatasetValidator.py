from deepdiff import DeepDiff
import numpy as np


class DatasetValidator():

    image_size = None
    masks_size = None

    def validate(self, image, masks, name):
        assert image is not None, "Got a None image for %s" % name
        assert masks is not None, "Got a None mask for %s" % name
        assert isinstance(
            image,
            np.ndarray), "Image is not an ndarray, it is %s." % type(image)
        assert isinstance(
            masks,
            np.ndarray), "Mask is not an ndarray, it is %s." % type(masks)
        assert image.dtype == 'float32', "Image should be of type float32"
        assert masks.dtype == 'float32', "Masks should be of type float32"
        assert np.count_nonzero(masks) == np.sum(
            masks), "Mask should only contain zero or one values."
        assert np.min(
            image
        ) >= 0., "Image should not contain values below 0.  Found {}".format(
            np.min(image))
        assert np.max(
            image
        ) <= 1., "Image should not contain values above 1.  Found {}".format(
            np.max(image))

        if self.image_size is None:
            self.image_size = image.shape
        if self.masks_size is None:
            self.masks_size = masks.shape
            for l in range(0, len(self.image_size)):
                assert self.image_size[l] == self.masks_size[
                    l], "Mask size does not match image size on axis %s." % l
        assert not DeepDiff(
            image.shape,
            self.image_size), "Image size is not consistent in %s" % name
        assert not DeepDiff(
            masks.shape,
            self.masks_size), "Mask size is not consistent in %s" % name
