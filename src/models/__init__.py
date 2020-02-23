from .unet import custom_unet as unet
from segmentation_models import Unet as segmentations_unet
from tensorflow.keras.regularizers import l1_l2
from activations import get_activation
from tensorflow.keras.layers import Input, Conv2D, Average
from tensorflow.keras.models import Model

import os

def find_latest_weight(folder):
    if not os.path.isdir(folder):
        return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")]
    if len(files) == 0:
        return None
    return max(files, key=os.path.getctime)

def find_best_weight(folder):
    if not os.path.isdir(folder):
        return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")]
    if len(files) == 0:
        return None
    return min(files, key=lambda x: float(x.split("-")[1]))

def get_model(image_size, job_config):

    regularizer = l1_l2(l1=job_config["L1_REG"], l2=job_config["L2_REG"])
    model = None

    if job_config["MODEL"]["NAME"] == "unet":
        model = unet(
            input_shape=image_size,
            use_batch_norm=job_config["BATCH_NORM"],
            filters=job_config["MODEL"]["FILTERS"],
            dropout=job_config["DROPOUT"],
            dropout_change_per_layer=job_config["DROPOUT_CHANGE_PER_LAYER"],
            use_dropout_on_upsampling=job_config["USE_DROPOUT_ON_UPSAMPLE"],
            activation=get_activation(job_config["MODEL"]["ACTIVATION"]),
            kernel_initializer='he_normal',
            num_layers=job_config["MODEL"]["LAYERS"]
        )

    if job_config["MODEL"]["NAME"] == "segmentations_unet":
        model = segmentations_unet(
            backbone_name=job_config["MODEL"]["BACKBONE"],
            encoder_weights=None,
            input_shape=image_size
        )

    if model is None:
        raise ValueError("Model %s not defined" % job_config["MODEL"]["NAME"])

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    return model

def model_for_folds(clazz, modeldir, job_config, job_hash, folds=None, load_weights=True):

    inputs = Input(shape=(None, None, 1))

    models = []

    if folds is None:
        folds = range(job_config["FOLDS"])

    for fold in folds:
        fold_model = get_model((None, None, 1), job_config)
        if load_weights:
            best_weight = find_best_weight( os.path.join(modeldir, job_hash, clazz, "fold%s" % fold))
            if best_weight is None:
                print("Could not find weight for fold %s - skipping" % fold)
                continue
            print("Loading weight %s" % best_weight)
            fold_model.load_weights(best_weight)
        fold_model.trainable = False
        fold_model._name = "fold%s" % fold

        out = fold_model(inputs)
        models.append(out)

    if len(models) == 0:
        print("No models found for class %s - skipping" % clazz)
        return None

    if len(models) == 1:
        out = models[0]
    else:
        out = Average()(models)

    model = Model(inputs, out)

    return model