import os
from segmenter.jobs import tasks

if os.environ.get("DEBUG", "false").lower() != "true":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
