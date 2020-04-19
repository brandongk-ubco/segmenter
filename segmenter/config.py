import os
import json
from segmenter.helpers import hash, get_available_gpus
from typing import Dict, Any

if os.environ.get("COMMAND", "train") == "evaluate":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
if os.environ.get("DEBUG", "false").lower() != "true":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_batch_size():
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
    return BATCH_SIZE


def get_classes(path):
    with open(os.path.join(path, "classes.json"), "r") as json_file:
        data = json.load(json_file)
        return data["class_order"]


def get_model():
    model = {"NAME": os.environ.get("MODEL", "segmentations_unet")}

    if model["NAME"] == "unet":
        model["FILTERS"] = int(os.environ.get("FILTERS", 4))
        model["ACTIVATION"] = os.environ.get("ACTIVATION", "cos")
        model["LAYERS"] = int(os.environ.get("LAYERS", 4))
        model["NORM"] = os.environ.get("NORM", "batch").lower()
        model["DROPOUT"] = float(os.environ.get("DROPOUT", 0.0))
        model["USE_DROPOUT_ON_UPSAMPLE"] = os.environ.get(
            "USE_DROPOUT_ON_UPSAMPLE", "true").lower() == "true"
        model["DROPOUT_CHANGE_PER_LAYER"] = float(
            os.environ.get("DROPOUT_CHANGE_PER_LAYER", 0.0))
        model["MAX_DROPOUT"] = float(os.environ.get("DROPOUT", 0.5))
        model["FILTER_RATIO"] = float(os.environ.get("FILTER_RATIO", 2))
    if model["NAME"] == "segmentations_unet":
        model["BACKBONE"] = os.environ.get("BACKBONE", "efficientnetb0")

    return model


def get_loss():
    return {
        "DICE_BETA":
        float(os.environ.get("DICE_BETA", 1)),
        "BFL_GAMMA":
        float(os.environ.get("BFL_GAMMA", 2)),
        "DICE_MULTIPLIER":
        float(os.environ.get("DICE_MULTIPLIER", 1)),
        "BFL_MULTIPLIER":
        float(os.environ.get("BFL_MULTIPLIER", 0)),
        "BCE_MULTIPLIER":
        float(os.environ.get("BCE_MULTIPLIER", 1)),
        "BCE_FROM_LOGITS":
        os.environ.get("BCE_FROM_LOGITS", "false ").lower() == "true"
    }


def get_augments():
    return {
        "RESCALE_PERCENTAGE": float(os.environ.get("RESCALE_PERCENTAGE",
                                                   0.15)),
        "RESCALE_PR": float(os.environ.get("RESCALE_PR", 0.0)),
        "ELASTIC_TRANSFORM_PR":
        float(os.environ.get("ELASTIC_TRANSFORM_PR", 0.0)),
        "GAMMA_PR": float(os.environ.get("GAMMA_PR", 0.0)),
        "HORIZONTAL_FLIP_PR": float(os.environ.get("HORIZONTAL_FLIP_PR", 0.5)),
        "VERTICAL_FLIP_PR": float(os.environ.get("VERTICAL_FLIP_PR", 0.5)),
        "NORMALIZE_PR": float(os.environ.get("NORMALIZE_PR", 0.0)),
    }


def get_preprocess():
    return {
        "ZERO_BLANKS": os.environ.get("ZERO_BLANKS",
                                      "false").lower() == "true",
    }


def get_postprocess():
    return {"RECENTER": os.environ.get("RECENTER", "false").lower() == "true"}


def get_optimizer():
    optimizer = {"NAME": os.environ.get("OPTIMIZER", "adam")}

    if optimizer["NAME"] == "adam":
        optimizer["BETA_1"] = float(os.environ.get("BETA_1", 0.9))
        optimizer["BETA_2"] = float(os.environ.get("BETA_2", 0.999))
        optimizer["AMSGRAD"] = os.environ.get("AMSGRAD",
                                              "true").lower() == "false"
        optimizer["LR"] = float(os.environ.get("LR", 0.001))

    if optimizer["NAME"] == "sgd":
        optimizer["LR"] = float(os.environ.get("LR", 0.001))
        optimizer["MOMENTUM"] = float(os.environ.get("MOMENTUM", 0.9))
        optimizer["NESTEROV"] = os.environ.get("NESTEROV",
                                               "false").lower() == "true"

    if optimizer["NAME"] == "adabound":
        optimizer["LR"] = float(os.environ.get("LR", 0.1))

    if optimizer["NAME"] == "combined":
        optimizer = [{
            "NAME": "adam",
            "BETA_1": float(os.environ.get("BETA_1", 0.9)),
            "BETA_2": float(os.environ.get("BETA_2", 0.999)),
            "AMSGRAD": os.environ.get("AMSGRAD", "true").lower() == "false",
            "LR": float(os.environ.get("LR", 0.001))
        }, {
            "NAME":
            "sgd",
            "LR":
            float(os.environ.get("LR", 0.001)),
            "MOMENTUM":
            float(os.environ.get("MOMENTUM", 0.5)),
            "NESTEROV":
            os.environ.get("NESTEROV", "false").lower() == "true"
        }]

    return optimizer


# WARNING when setting precision!
#  - larger images need higher precision to evaluate the Loss function.
#    The sum can overflow and give all 0 for loss.
def build_config(datadir: str):
    return {
        "BATCH_SIZE": get_batch_size(),
        "PATIENCE": int(os.environ.get("PATIENCE", 20)),
        "MIN_LR": float(os.environ.get("MIN_LR", 1e-8)),
        "LR_REDUCTION_FACTOR": float(os.environ.get("LR_REDUCTION_FACTOR",
                                                    0.1)),
        "L1_REG": float(os.environ.get("L1_REG", 0)),
        "L2_REG": float(os.environ.get("L2_REG", 0)),
        "OPTIMIZER": get_optimizer(),
        "FSCORE_THRESHOLD": float(os.environ.get("FSCORE_THRESHOLD", 0.5)),
        "BOOST_FOLDS": int(os.environ.get("BOOST_FOLDS", 0)),
        "FOLDS": int(os.environ.get("FOLDS", 0)),
        "CLASSES": get_classes(datadir),
        "MODEL": get_model(),
        "PRECISION": os.environ.get("PRECISION", "float32"),
        "LOSS": get_loss(),
        "AUGMENTS": get_augments(),
        "PREPROCESS": get_preprocess(),
        "POSTPROCESS": get_postprocess(),
        "RUN": int(os.environ.get("RUN", 1)),
    }


def get_config(datadir: str, outdir: str):
    job_config: Dict[str, Any] = {}

    if "JOB_HASH" not in os.environ:
        job_config = build_config(datadir)
        job_hash = hash(job_config)
    else:
        job_hash = os.environ["JOB_HASH"]
        config_location = os.path.join(outdir, job_hash, "config.json")
        if os.path.isfile(config_location):
            with open(config_location, "r") as config_file:
                job_config = json.load(config_file)
        else:
            job_config = build_config(datadir)
        assert hash(
            job_config
        ) == job_hash, "Expected job hash ({}) doesn't match actual ({})".format(
            job_hash, hash(job_config))

    os.environ["JOB_HASH"] = job_hash
    return job_config, job_hash


def validate_config(config: Dict[str, Any]):
    num_gpus = get_available_gpus()
    assert num_gpus == 0 or config["BATCH_SIZE"] == int(
        config["BATCH_SIZE"] / num_gpus
    ) * num_gpus, "Batch size must be an integer multiple of the number of gpus ({}).".format(
        num_gpus)