from tensorflow.keras.optimizers import Adam, SGD
from .AdaBound import AdaBound

def get_optimizer(optimizer_config):

    if isinstance(optimizer_config, list):
        return [get_optimizer(o) for o in optimizer_config]
    
    if optimizer_config["NAME"] == "adam":
        return Adam(
            learning_rate=optimizer_config["LR"],
            beta_1=optimizer_config["BETA_1"],
            beta_2=optimizer_config["BETA_2"],
            amsgrad=optimizer_config["AMSGRAD"]
        )

    if optimizer_config["NAME"] == "sgd":
        return SGD(
            learning_rate=optimizer_config["LR"],
            momentum=optimizer_config["MOMENTUM"],
            nesterov=optimizer_config["NESTEROV"]
        )

    if optimizer_config["NAME"] == "adabound":
        return SGD(
            learning_rate=optimizer_config["LR"]
        )