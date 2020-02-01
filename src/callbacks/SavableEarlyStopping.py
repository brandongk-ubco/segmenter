from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

import os
import json

class SavableEarlyStopping(EarlyStopping):

    def __init__(self, stateFile, **kwargs):
        super(SavableEarlyStopping, self).__init__(**kwargs)
        self.stateFile = stateFile

    def on_epoch_end(self, epoch, logs=None):
        super(SavableEarlyStopping, self).on_epoch_end(epoch, logs)
        self.save()

    def on_train_begin(self, logs=None):
        super(SavableEarlyStopping, self).on_train_begin(logs)
        self.restore()

    def save(self):
        state = {
            "monitor": self.monitor,
            "baseline": self.baseline,
            "best": self.best,
            "patience": self.patience,
            "verbose": self.verbose,
            "min_delta": self.min_delta,
            "wait": self.wait,
            "stopped_epoch": self.stopped_epoch,
            "restore_best_weights": self.restore_best_weights,
            "best_weights": self.best_weights,
            "monitor_op": str(self.monitor_op)
        }

        with open(self.stateFile, 'w') as state_json:
            json.dump(state, state_json, indent=4)


    def restore(self):
        if not os.path.isfile(self.stateFile):
            return

        with open(self.stateFile, 'r') as state_json:
            state = json.load(state_json)

        self.monitor = state["monitor"]
        self.best = state["best"]
        self.baseline = state["best"]
        self.patience = state["patience"]
        self.verbose = state["verbose"]
        self.min_delta = state["min_delta"]
        self.wait = state["wait"]
        self.stopped_epoch = state["stopped_epoch"]
        self.restore_best_weights = state["restore_best_weights"]
        self.best_weights = state["best_weights"]

        if str(state["monitor_op"]) == str(np.greater):
            self.monitor_op = np.greater

        if str(state["monitor_op"]) == str(np.less):
            self.monitor_op = np.less
