from segmenter.helpers.get_available_gpus import get_available_gpus
from typing import Dict, Any


def validate_config(config: Dict[str, Any]):
    num_gpus = get_available_gpus()
    assert num_gpus == 0 or config["BATCH_SIZE"] == int(
        config["BATCH_SIZE"] / num_gpus
    ) * num_gpus, "Batch size must be an integer multiple of the number of gpus ({}).".format(
        num_gpus)