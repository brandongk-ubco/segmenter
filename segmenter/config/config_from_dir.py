import os
import json
from typing import Dict, Any
from segmenter.config.hash_config import hash_config


def config_from_dir(directory: str):
    job_config: Dict[str, Any] = {}

    config_location = os.path.join(directory, os.environ["JOB_CONFIG"],
                                   "config.json")
    with open(config_location, "r") as config_file:
        job_config = json.load(config_file)
    job_hash = hash_config(job_config)

    assert job_hash == os.environ[
        "JOB_CONFIG"], "Expected job hash ({}) doesn't match actual ({})".format(
            job_hash, hash(job_config))

    return job_config, job_hash
