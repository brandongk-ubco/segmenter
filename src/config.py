import os
import json

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
    num_folds = int(os.environ.get("NUM_FOLDS", 10))
    train_behind = int(os.environ.get("TRAIN_BEHIND", 1))

    with open(os.path.join(os.environ.get("DATA_PATH", "/data"), "%s-trainbehind-%s-folds.json" % (train_behind, num_folds)), "r") as json_file:
        data = json.load(json_file)
        return data["class_order"]

def get_config():
    return {
        "BATCH_SIZE": get_batch_size(),
        "PATIENCE": int(os.environ.get("PATIENCE", 20)),
        "MIN_LR": float(os.environ.get("MIN_LR", 1e-8)),
        "DROPOUT": float(os.environ.get("DROPOUT", 0.0)),
        "USE_DROPOUT_ON_UPSAMPLE": os.environ.get("USE_DROPOUT_ON_UPSAMPLE", "true").lower() == "true",
        "DROPOUT_CHANGE_PER_LAYER": float(os.environ.get("DROPOUT_CHANGE_PER_LAYER", 0.0)),
        "LR_REDUCTION_FACTOR": float(os.environ.get("LR_REDUCTION_FACTOR", 0.1)),
        "ACTIVATION": os.environ.get("ACTIVATION", "cos"),
        "FILTERS": int(os.environ.get("FILTERS", 16)),
        "L1_REG": float(os.environ.get("L1_REG", 3e-5)),
        "L2_REG": float(os.environ.get("L2_REG", 3e-5)),
        "LR": float(os.environ.get("LR", 0.001)),
        "BETA_1": float(os.environ.get("BETA_1", 0.9)),
        "BETA_2": float(os.environ.get("BETA_2", 0.999)),
        "AMSGRAD": os.environ.get("AMSGRAD", "true").lower() == "true",
        "BATCH_NORM": os.environ.get("BATCH_NORM", "true").lower() == "true",
        "ELASTIC_TRANSFORM_PR": float(os.environ.get("ELASTIC_TRANSFORM_PR", 0.0)),
        "LAYERS": int(os.environ.get("LAYERS", 4)),
        "FSCORE_THRESHOLD": float(os.environ.get("FSCORE_THRESHOLD", 0.5)),
        "TRAIN_BEHIND": int(os.environ.get("TRAIN_BEHIND", 1)),
        "NUM_FOLDS": int(os.environ.get("NUM_FOLDS", 10)),
        "CLASSES": get_classes(),
        "RESCALE_PERCENTAGE": float(os.environ.get("RESCALE_PERCENTAGE", 0.15)),
    }