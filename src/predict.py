from config import get_config

import pprint

from helpers import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from metrics import get_metrics

from models import model_for_folds
from augmentations import predict_augments
from loss import get_loss

from DataGenerator import DataGenerator

from callbacks import get_evaluation_callbacks
import json
from matplotlib import pyplot as plt

job_config = get_config()
job_hash = hash(job_config)

outdir = os.environ.get("DIRECTORY", "/output")
outdir = os.path.abspath(outdir)

def evaluate(clazz, folds=None, method="include", overwrite=False):
    K.clear_session()

    generators, dataset, num_images = generate_for_augments(clazz, None, predict_augments, job_config, method=method, mode="evaluate")
    dataset = dataset.batch(1, drop_remainder=True)

    if folds is None:
        folds = range(job_config["FOLDS"])

    model_dir = os.path.join(outdir, job_hash, clazz, "results", "-".join([str("fold%s" % f) for f in folds]))

    threshold = job_config["FSCORE_THRESHOLD"]
    loss = get_loss(job_config["LOSS"])
    metrics = get_metrics(threshold, job_config["LOSS"])

    if not os.path.isdir(model_dir) or overwrite:
        print("Creating new model for %s" % model_dir)
        model = model_for_folds(clazz, outdir, job_config, job_hash, folds=folds)
    else:
        print("Loading existing model from %s" % model_dir)
        model = model_for_folds(clazz, outdir, job_config, job_hash, folds=folds, load_weights=False)
        model.load_weights(os.path.join(model_dir, "weights.h5"))

    model.compile(
        optimizer=Adam(
            learning_rate=job_config["LR"],
            beta_1=job_config["BETA_1"],
            beta_2=job_config["BETA_2"],
            amsgrad=job_config["AMSGRAD"]
        ),
        loss=loss,
        metrics=list(metrics.values())
    )

    if not os.path.isdir(model_dir) or overwrite:
        os.makedirs(model_dir, exist_ok=True)
        model.save_weights(os.path.join(model_dir, "weights.h5"))

    for batch, (images, masks) in enumerate(dataset):
        predictions = model.predict_on_batch(images).numpy()
        for i in range(predictions.shape[0]):
            prediction = predictions[i]
            mask = masks[i].numpy()
            image = images[i].numpy()
            prediction = prediction[:,:,0]
            mask = mask[:,:,0]
            if image.shape[2] == 1:
                image = image[:,:,0]
            plt.imshow(mask, cmap='gray')
            plt.savefig(os.path.join(model_dir, "%s-%s-mask.png" % (batch, i)))
            plt.imshow(image, cmap='gray')
            plt.savefig(os.path.join(model_dir, "%s-%s-image.png" % (batch, i)))

            for threshold in np.linspace(0.05, 0.95, num=19):
                thresholded = np.where(prediction > threshold, 1, 0).astype(prediction.dtype)
                plt.imshow(thresholded, cmap='gray')
                plt.savefig(os.path.join(model_dir, "%s-%s-%0.2f-threshold.png" % (batch, i, threshold)))
    import pdb
    pdb.set_trace()
    return

if __name__ == "__main__":
    pprint.pprint(job_config)

    classes = job_config["CLASSES"] if os.environ.get("CLASS") is None else [os.environ.get("CLASS")]

    if os.environ.get("EVAL_FOLDS") is not None:
        folds = [int(f.strip()) for f in os.environ.get("EVAL_FOLDS").split(",")]
    elif os.environ.get("FOLD") is not None:
        folds = [int(os.environ.get("FOLD").strip())]
    else:
        folds = range(job_config["FOLDS"])

    folds = list(folds)
    print("Predicting classes %s and folds %s" % (classes, folds))
    evaluations = {}
    for clazz in classes:
        evaluations[clazz] = {}
        # print("In-Class evaluation for %s" % clazz)
        # evaluations[clazz]["in_class"] = evaluate(clazz, folds=folds, method="include")
        # print("Out-Of-Class evaluation for %s" % clazz)
        evaluations[clazz]["in_class"] = evaluate(clazz, folds=folds)
    pprint.pprint(evaluations)
