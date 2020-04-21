from segmenter.models.unet import custom_unet as unet
from segmentation_models import Unet as segmentations_unet
from segmenter.activations import get_activation
import tensorflow as tf
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Lambda, Activation, Add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.activations import linear, sigmoid
from tensorflow.keras import backend as K
import os
from segmenter.helpers import logit
from segmenter.aggregators import Aggregator


def find_latest_weight(folder):
    if not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")
    ]
    if len(files) == 0:
        return None
    return max(files, key=os.path.getctime)


def find_best_weight(folder):
    if not os.path.isdir(folder):
        return None
    files = [
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) if f.endswith(".h5")
    ]
    if len(files) == 0:
        return None
    return min(files, key=lambda x: float(x.split("-")[1][:-3]))


def get_model(image_size, job_config):

    regularizer = l1_l2(l1=job_config["L1_REG"], l2=job_config["L2_REG"])
    model = None

    if job_config["MODEL"]["NAME"] == "unet":
        model = unet(input_shape=image_size,
                     norm=job_config["MODEL"]["NORM"],
                     filters=job_config["MODEL"]["FILTERS"],
                     dropout=job_config["MODEL"]["DROPOUT"],
                     dropout_change_per_layer=job_config["MODEL"]
                     ["DROPOUT_CHANGE_PER_LAYER"],
                     use_dropout_on_upsampling=job_config["MODEL"]
                     ["USE_DROPOUT_ON_UPSAMPLE"],
                     activation=get_activation(
                         job_config["MODEL"]["ACTIVATION"]),
                     kernel_initializer='he_normal',
                     num_layers=job_config["MODEL"]["LAYERS"],
                     max_dropout=job_config["MODEL"]["MAX_DROPOUT"],
                     filter_ratio=job_config["MODEL"]["FILTER_RATIO"])

    if job_config["MODEL"]["NAME"] == "segmentations_unet":
        model = segmentations_unet(
            backbone_name=job_config["MODEL"]["BACKBONE"],
            encoder_weights=None,
            input_shape=image_size)

    if model is None:
        raise ValueError("Model %s not defined" % job_config["MODEL"]["NAME"])

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    return model


def model_folds(inputs, clazz, modeldir, job_config, job_hash,
                fold_activation):
    models = []

    folds = range(job_config["FOLDS"]) if job_config["FOLDS"] else [None]
    boost_folds = range(job_config["BOOST_FOLDS"] +
                        1) if job_config["BOOST_FOLDS"] is not None else [
                            None
                        ]

    for fold in folds:
        fold_models = []
        fold_name = "all" if fold is None else "fold{}".format(fold)
        for boost_fold in boost_folds:
            boost_fold_name = fold_name
            boost_fold_name += "" if boost_fold is None else "b{}".format(
                boost_fold)

            best_weight = find_best_weight(
                os.path.join(modeldir, job_hash, clazz, boost_fold_name))
            assert best_weight is not None, "Could not find weight for fold %s" % fold

            boost_fold_model = get_model((None, None, 1), job_config)
            print("Loading weight %s" % best_weight)
            boost_fold_model.load_weights(best_weight)
            boost_fold_model.trainable = False
            boost_fold_model._name = boost_fold_name
            boost_fold_model = boost_fold_model(inputs)
            boost_fold_model = Lambda(
                lambda x: logit(x),
                name="{}_logit".format(boost_fold_name))(boost_fold_model)

            fold_models.append(boost_fold_model)

        fold_model = fold_models[0] if len(fold_models) == 1 else Add(
            name="{}_add".format(fold_name))(fold_models)

        fold_activation_name = "{}_{}".format(fold_name, fold_activation)
        fold_model = Activation(fold_activation,
                                name=fold_activation_name)(fold_model)
        model = Model(inputs, fold_model)
        model._name = fold_name
        models.append(model)
    return models


def full_model(clazz, modeldir, job_config, job_hash, aggregator):

    inputs = Input(shape=(None, None, 1))

    weight_file = os.path.join(modeldir, job_hash, clazz,
                               "{}.h5".format(aggregator.name()))
    # model_location = os.path.join(modeldir, job_hash, clazz,
    #                               "{}".format(aggregator.name()))

    # if os.path.exists(model_location):
    #     return load_model(model_location)
    models = model_folds(inputs, clazz, modeldir, job_config, job_hash,
                         aggregator.fold_activation())

    assert len(models) > 0, "No models found for class %s" % clazz

    if len(models) == 1:
        out = models[0](inputs)
    else:
        out = aggregator.layer()(name="aggregator")(
            [m(inputs) for m in models])
    out = Activation(aggregator.final_activation(),
                     name=aggregator.final_activation())(out)
    model = Model(inputs, out)
    model._name = "{}_{}".format(clazz, aggregator.name())
    model.summary()

    print("Saving Weights")
    model.save_weights(weight_file)
    # TODO: Saving and loading models in Keras is really hard, apparently...
    # print("Saving Model")
    # with open(model_location, "w") as model_file:
    #     model_file.write(model.to_json(indent=4))

    return model
