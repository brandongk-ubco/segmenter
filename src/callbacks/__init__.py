from .EarlyStoppingByTime import EarlyStoppingByTime
from .SavableEarlyStopping import SavableEarlyStopping
from .SavableReduceLROnPlateau import SavableReduceLROnPlateau
from .OptimizerSaver import OptimizerSaver
from .SubModelCheckpoint import SubModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard, TerminateOnNaN, TerminateOnNaN, CSVLogger, LambdaCallback

import os

def get_prediction_callbacks():
    return []

def get_evaluation_callbacks():
    return []

def get_callbacks(output_folder, job_config, fold, val_loss, start_time, fold_name, loss, metrics):

    log_folder = os.path.join(output_folder, "logs")
    os.makedirs(log_folder, exist_ok=True)

    lr_reducer = SavableReduceLROnPlateau(
        os.path.join(output_folder, "lr_reducer.json"),
        factor=job_config["LR_REDUCTION_FACTOR"],
        cooldown=job_config["PATIENCE"],
        patience=job_config["PATIENCE"],
        min_lr=job_config["MIN_LR"],
        monitor='val_loss',
        mode='min',
        verbose=2
    )

    model_autosave = SubModelCheckpoint(
        filepath=os.path.join(output_folder, "{epoch:04d}-{val_loss:.6f}.h5"),
        submodel=fold_name,
        save_best_only=False,
        save_weights_only=True
    )
    model_autosave.best = val_loss

    early_stopping = SavableEarlyStopping(
        os.path.join(output_folder, "early_stopping.json"),
        patience=job_config["PATIENCE"]*3,
        verbose=2,
        monitor='val_loss',
        mode='min'
    )

    tensorboard = TensorBoard(
        log_dir=os.path.join(log_folder, "tenboard"),
        profile_batch=0
    )

    logger = CSVLogger(
        os.path.join(log_folder, 'train.csv'),
        separator=',',
        append=True
    )

    time_limit = EarlyStoppingByTime(
        limit_seconds=int(os.environ.get("LIMIT_SECONDS", -1)),
        start_time=start_time,
        verbose=0
    )

    optimizer_saver = OptimizerSaver(
        os.path.join(output_folder, "optimizer.pkl"),
        optimizer_config=job_config["OPTIMIZER"],
        loss=loss,
        metrics=metrics
    )

    return [optimizer_saver, lr_reducer, TerminateOnNaN(), early_stopping, logger, tensorboard, model_autosave, time_limit]