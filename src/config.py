from tensorflow.python.client import device_lib
import os

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

def get_batch_size():
    # This needs to be evenly distributed by the number of GPUs, otherwise you get NaN losses
    num_gpus = get_available_gpus()
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
    return int(BATCH_SIZE / num_gpus) * num_gpus

def get_config():
    return {
        "BATCH_SIZE": get_batch_size(),
        "PATIENCE": int(os.environ.get("PATIENCE", 20)),
        "MIN_LR": float(os.environ.get("MIN_LR", 1e-8)),
        "DROPOUT": float(os.environ.get("DROPOUT", 0.1)),
        "USE_DROPOUT_ON_UPSAMPLE": os.environ.get("AMSGRAD", "true").lower() == "true",
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
        "ELASTIC_TRANSFORM_PR": float(os.environ.get("ELASTIC_TRANSFORM_PR", 0.5)),
        "LAYERS": int(os.environ.get("LAYERS", 4)),
    }