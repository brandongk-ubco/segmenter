from tensorflow.keras import backend as K
from segmenter.evaluators.ThresholdAwareEvaluator import ThresholdAwareEvaluator
import numpy as np
import os


class PredictEvaluator(ThresholdAwareEvaluator):
    def evaluate_threshold(self, model, threshold, outdir):
        for batch, (images, masks) in enumerate(self.dataset):
            name = os.path.basename(self.generator.image_files[batch])
            print("{} ({}/{})".format(name, batch, self.num_images))
            predictions = model.predict_on_batch(images).numpy()
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

                intr = np.sum(np.logical_and(boolean_prediction, boolean_mask))
                union = np.sum(np.logical_or(boolean_prediction, boolean_mask))
                iou = intr / union

                name = "prediction-{}-{:.4f}".format(name, iou)

                np.savez_compressed(os.path.join(outdir, name),
                                    image=image,
                                    prediction=thresholded_prediction,
                                    mask=mask)
