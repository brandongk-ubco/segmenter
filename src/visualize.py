from config import get_config

import pprint

from helpers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from metrics import get_metrics
from models import full_model
from augmentations import predict_augments
from loss import get_loss
from DataGenerator import DataGenerator
from aggregators import get_aggregators
from callbacks import get_evaluation_callbacks
import json
from matplotlib import pyplot as plt
from optimizers import get_optimizer

job_config = get_config()
job_hash = hash(job_config)

outdir = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(outdir)

def evaluate(clazz):
    K.clear_session()

    generators, dataset, num_images = generate_for_augments(clazz, None, predict_augments, job_config, mode="evaluate")
    dataset = dataset.batch(1, drop_remainder=True)

    for aggregator_name, aggregator, fold_activation, final_activation, thresholds in get_aggregators(job_config):
        model_dir = os.path.join(outdir, job_hash, clazz, "results", aggregator_name)
        loss = get_loss(job_config["LOSS"])

        print("Creating model for %s" % model_dir)
        model = full_model(clazz, outdir, job_config, job_hash, aggregator=aggregator, fold_activation=fold_activation, final_activation=final_activation)
        os.makedirs(model_dir, exist_ok=True)
        model.save_weights(os.path.join(model_dir, "weights.h5"))

        results = {}
        for threshold in thresholds:
            threshold_str = "{:1.2f}".format(threshold)
            threshold_dir = os.path.join(model_dir, threshold_str)
            os.makedirs(threshold_dir, exist_ok=True)

            metrics = get_metrics(threshold, job_config["LOSS"])
            model.compile(
                optimizer=get_optimizer(job_config["OPTIMIZER"]),
                loss=loss,
                metrics=list(metrics.values())
            )

            for batch, (images, masks) in enumerate(dataset):
                print("Aggregator {} and Threshold: {} - {}/{}".format(aggregator_name, threshold_str, batch, num_images))
                predictions = model.predict_on_batch(images).numpy()
                for i in range(predictions.shape[0]):
                    prediction = predictions[i]
                    mask = masks[i].numpy()
                    image = images[i].numpy()
                    prediction = prediction[:,:,0]
                    mask = mask[:,:,0]
                    thresholded = np.where(prediction > threshold, 1, 0).astype(prediction.dtype)
                    if image.shape[2] == 1:
                        image = image[:,:,0]

                    boolean_mask = mask.astype(bool)
                    boolean_prediction= thresholded.astype(bool)

                    highlighted_image = np.stack([image.copy(), image.copy(), image.copy()], axis=len(image.shape))

                    highlighted_mask = np.zeros(highlighted_image.shape, dtype=highlighted_image.dtype)

                    #False Positives (red)
                    highlighted_mask[:,:,0][np.logical_and(np.logical_xor(boolean_mask, boolean_prediction), boolean_prediction)] = 1. 

                    #True Positives (green)
                    highlighted_mask[:,:,1][np.logical_and(boolean_mask, boolean_prediction)] = 1.

                    #False Negatives (blue)
                    highlighted_mask[:,:,2][np.logical_and(np.logical_xor(boolean_mask, boolean_prediction), np.logical_not(boolean_prediction))] = 1.

                    iou = np.sum(np.logical_and(boolean_prediction, boolean_mask)) / np.sum(np.logical_or(boolean_prediction, boolean_mask))
                    name = "{}-{:.4f}".format(batch, iou)

                    highlighted_image[highlighted_mask == 1] = 1.

                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

                    fig.set_size_inches(10, 5)

                    ax1.imshow(image, cmap='gray')
                    ax1.axis('off')
                    ax1.set_title('Original Image')

                    ax2.imshow(mask, cmap='gray')
                    ax2.axis('off')
                    ax2.set_title('Original Mask')

                    ax3.imshow(highlighted_image)
                    ax3.axis('off')
                    ax3.set_title('Predictions')

                    ax4.imshow(highlighted_mask, cmap='gray')
                    ax4.set_title("Predicted Mask (IOU {:.2f})".format(iou))
                    ax4.axis('off')

                    plt.subplots_adjust(wspace=0, hspace=0)

                    plt.savefig(os.path.join(threshold_dir, "{}.png".format(name)), dpi=100, bbox_inches = 'tight', pad_inches = 0.5)
                    plt.close()

                    np.savez_compressed(os.path.join(threshold_dir, name), image=image, mask=mask, highlighted_image=highlighted_image, highlighted_mask=highlighted_mask)


if __name__ == "__main__":
    pprint.pprint(job_config)

    classes = job_config["CLASSES"] if os.environ.get("CLASS") is None else [os.environ.get("CLASS")]
    print("Visualizing classes %s" % (classes))
    for clazz in classes:
        evaluate(clazz)
