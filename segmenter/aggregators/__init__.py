from .LeakyNoisyOr import LeakyNoisyOr
from .Vote import Vote
from tensorflow.keras.layers import Average
import numpy as np

def get_aggregators(job_config, aggregators=None):
    num_folds = job_config.get("FOLDS", 0)
    ag = [
        ("vote", Vote, "sigmoid", "linear", [0] if num_folds == 0 else np.linspace(0., (num_folds - 1) / num_folds, num=num_folds)),
        ("leaky_noisy_or", LeakyNoisyOr, "sigmoid", "linear", np.append(np.linspace(0.5, 0.75, num=6), np.linspace(0.8, 0.99, num=20))),
        ("average", Average, "linear", "sigmoid", np.linspace(0., 0.95, num=20))
    ]
    if aggregators is not None:
        ag = list(filter(lambda x: x[0] in aggregators, ag))
    return ag