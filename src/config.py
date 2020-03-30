import os
import json

if os.environ.get("COMMAND", "train") == "evaluate":
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
if os.environ.get("DEBUG", "false").lower() != "true":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

def get_available_gpus():
    try:
        from tensorflow.python.client import device_lib
        local_device_protos = device_lib.list_local_devices()
        return len([x.name for x in local_device_protos if x.device_type == 'GPU'])
    except ModuleNotFoundError:
        return 0

def get_batch_size():
    # This needs to be evenly distributed by the number of GPUs, otherwise you get NaN losses
    num_gpus = get_available_gpus()
    if num_gpus == 0:
        return 1
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
    return int(BATCH_SIZE / num_gpus) * num_gpus

def get_classes():
    folds = int(os.environ.get("FOLDS", 10))
    boost_folds = int(os.environ.get("BOOST_FOLDS", 0))
    path = os.path.abspath(os.environ.get("DATA_PATH", "/data"))

    with open(os.path.join(path, "classes.json"), "r") as json_file:
        data = json.load(json_file)
        return data["class_order"]

def get_model():
    model = {
        "NAME": os.environ.get("MODEL", "unet")
    }

    if model["NAME"] == "unet":
        model["FILTERS"] = int(os.environ.get("FILTERS", 16))
        model["ACTIVATION"] = os.environ.get("ACTIVATION", "cos")
        model["LAYERS"] = int(os.environ.get("LAYERS", 4))
        model["BATCH_NORM"] = os.environ.get("BATCH_NORM", "true").lower() == "true"
        model["DROPOUT"] = float(os.environ.get("DROPOUT", 0.0))
        model["USE_DROPOUT_ON_UPSAMPLE"] = os.environ.get("USE_DROPOUT_ON_UPSAMPLE", "true").lower() == "true"
        model["DROPOUT_CHANGE_PER_LAYER"] = float(os.environ.get("DROPOUT_CHANGE_PER_LAYER", 0.0))
        model["MAX_DROPOUT"] = float(os.environ.get("DROPOUT", 0.5))
    if model["NAME"] == "segmentations_unet":
        model["BACKBONE"] = os.environ.get("BACKBONE", "efficientnetb0")

    return model

def get_loss():
    return {
        "DICE_BETA": float(os.environ.get("DICE_BETA", 1)),
        "BFL_GAMMA": float(os.environ.get("BFL_GAMMA", 2)),
        "DICE_MULTIPLIER": float(os.environ.get("DICE_MULTIPLIER", 1)),
        "BFL_MULTIPLIER": float(os.environ.get("BFL_MULTIPLIER", 0)),
        "BCE_MULTIPLIER": float(os.environ.get("BCE_MULTIPLIER", 1)),
        "BCE_FROM_LOGITS": os.environ.get("BCE_FROM_LOGITS", "false ").lower() == "true"
    }

def get_augments():
    return {
        "RESCALE_PERCENTAGE": float(os.environ.get("RESCALE_PERCENTAGE", 0.15)),
        "RESCALE_PR": float(os.environ.get("RESCALE_PR", 0.0)),
        "ELASTIC_TRANSFORM_PR": float(os.environ.get("ELASTIC_TRANSFORM_PR", 0.0)),
        "GAMMA_PR": float(os.environ.get("GAMMA_PR", 0.0)),
        "HORIZONTAL_FLIP_PR": float(os.environ.get("HORIZONTAL_FLIP_PR", 0.5)),
        "VERTICAL_FLIP_PR": float(os.environ.get("VERTICAL_FLIP_PR", 0.5)),
        "NORMALIZE_PR": float(os.environ.get("NORMALIZE_PR", 1.0)),
    }

def get_preprocess():
    return {
        "ZERO_BLANKS": os.environ.get("RECENTER", "true").lower() == "true",
    }

def get_postprocess():
    return {
        "RECENTER": os.environ.get("RECENTER", "false").lower() == "true"
    }

def get_optimizer():
    return {
        "NAME": "adam",
        "BETA_1": float(os.environ.get("BETA_1", 0.9)),
        "BETA_2": float(os.environ.get("BETA_2", 0.999)),
        "AMSGRAD": os.environ.get("AMSGRAD", "true").lower() == "true",
        "LR": float(os.environ.get("LR", 0.001)),
    }

# WARNING when setting precision - larger images need higher precision to evaluate the Loss function.  The sum can overflow and give all 0 for loss.
def get_config(path="/data"):
    return {
        "BATCH_SIZE": get_batch_size(),
        "PATIENCE": int(os.environ.get("PATIENCE", 20)),
        "MIN_LR": float(os.environ.get("MIN_LR", 1e-8)),
        "LR_REDUCTION_FACTOR": float(os.environ.get("LR_REDUCTION_FACTOR", 0.1)),
        "L1_REG": float(os.environ.get("L1_REG", 3e-5)),
        "L2_REG": float(os.environ.get("L2_REG", 3e-5)),
        "OPTIMIZER": get_optimizer(),
        "FSCORE_THRESHOLD": float(os.environ.get("FSCORE_THRESHOLD", 0.5)),
        "BOOST_FOLDS": None if os.environ.get("BOOST_FOLDS") is None else int(os.environ.get("BOOST_FOLDS", 0)),
        "FOLDS": None if os.environ.get("FOLDS") is None else int(os.environ.get("FOLDS", 0)),
        "CLASSES": get_classes(),
        "MODEL": get_model(),
        "PRECISION": os.environ.get("PRECISION", "float32"),
        "LOSS": get_loss(),
        "AUGMENTS": get_augments(),
        "PREPROCESS": get_preprocess(),
        "POSTPROCESS": get_postprocess(),
        "RUN": int(os.environ.get("RUN", 1)),
    }