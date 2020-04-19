from tensorflow.keras import backend as K
from segmenter.evaluators.BaseEvaluator import BaseEvaluator
import numpy as np
import os


class PredictEvaluator(BaseEvaluator):
    def evaluate_threshold(self, threshold, outdir):
        for batch, (images, masks) in enumerate(self.dataset):
            name = os.path.basename(self.generator.image_files[batch])
            print("{} ({}/{})".format(name, batch, self.num_images))
            predictions = self.model.predict_on_batch(images).numpy()
            for i in range(predictions.shape[0]):
                prediction = predictions[i]
                mask = masks[i].numpy()
                image = images[i].numpy()
                prediction = prediction[:, :, 0]
                mask = mask[:, :, 0]
                # TODO: Should this be > or >=?  Needs to match what the metric evaluation does.
                thresholded_prediction = np.where(prediction > threshold, 1,
                                                  0).astype(prediction.dtype)

                if image.shape[2] == 1:
                    image = image[:, :, 0]

                boolean_mask = mask.astype(bool)
                boolean_prediction = thresholded_prediction.astype(bool)

                # False Positives
                fp = np.zeros(image.shape, dtype=image.dtype)
                fp[np.logical_and(
                    np.logical_xor(boolean_mask, boolean_prediction),
                    boolean_prediction)] = 1.

                # True Positives
                tp = np.zeros(image.shape, dtype=image.dtype)
                tp[np.logical_and(boolean_mask, boolean_prediction)] = 1.

                # False Negatives
                fn = np.zeros(image.shape, dtype=image.dtype)
                fn[np.logical_and(
                    np.logical_xor(boolean_mask, boolean_prediction),
                    np.logical_not(boolean_prediction))] = 1.

                intr = np.sum(np.logical_and(boolean_prediction, boolean_mask))
                union = np.sum(np.logical_or(boolean_prediction, boolean_mask))

                iou = intr / union
                name = "{}-{:.4f}".format(name, iou)

                np.savez_compressed(
                    os.path.join(outdir, name),
                    image=image,
                    prediction=prediction,
                    thresholded_prediction=thresholded_prediction,
                    mask=mask,
                    false_positives=fp,
                    true_positives=tp,
                    false_negatives=fn)
