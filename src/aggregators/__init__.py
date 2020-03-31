from .LeakyNoisyOr import LeakyNoisyOr
from .Vote import Vote
from tensorflow.keras.layers import Average
import numpy as np

def get_aggregators(job_config):
    return [
        ("vote", Vote, "sigmoid", "linear", np.linspace(0., 1., num=job_config.get("FOLDS", 0) + 1)),
        ("leaky_noisy_or", LeakyNoisyOr, "sigmoid", "linear", np.linspace(0., 1., num=21)),
        ("average", Average, "linear", "sigmoid", np.linspace(0., 1., num=21))
    ]