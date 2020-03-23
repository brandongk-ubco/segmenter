from tensorflow.keras.optimizers import Adam

def get_optimizer(optimizer_config, boost_folds=None):
    lr_multiplier = 1
    if boost_folds is not None:
        lr_multiplier = (1 + len(boost_folds))
    if optimizer_config["NAME"] == "adam":
        return Adam(
            learning_rate=optimizer_config["LR"] * lr_multiplier,
            beta_1=optimizer_config["BETA_1"],
            beta_2=optimizer_config["BETA_2"],
            amsgrad=optimizer_config["AMSGRAD"]
        )