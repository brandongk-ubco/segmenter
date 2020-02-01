from tensorflow.keras import backend
from tensorflow.keras.callbacks import ReduceLROnPlateau

import os
import json

class SavableReduceLROnPlateau(ReduceLROnPlateau):

    def __init__(self, stateFile, **kwargs):
        super(SavableReduceLROnPlateau, self).__init__(**kwargs)
        self.stateFile = stateFile

    def on_epoch_end(self, epoch, logs=None):
        super(SavableReduceLROnPlateau, self).on_epoch_end(epoch, logs)
        self.save()

    def on_train_begin(self, logs=None):
        super(SavableReduceLROnPlateau, self).on_train_begin(logs)
        self.restore()

    def save(self):
        state = {
            "factor": self.factor,
            "min_lr": self.min_lr,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "verbose": self.verbose,
            "cooldown": self.cooldown,
            "cooldown_counter": self.cooldown_counter,
            "wait": self.wait,
            "best": self.best,
            "mode": self.mode,
            "lr": float(backend.get_value(self.model.optimizer.lr))
        }

        with open(self.stateFile, 'w') as state_json:
            json.dump(state, state_json, indent=4)

    def restore(self):

        if not os.path.isfile(self.stateFile):
            return

        with open(self.stateFile, 'r') as state_json:
            state = json.load(state_json)

        self.factor = state["factor"]
        self.min_lr = state["min_lr"]
        self.min_delta = state["min_delta"]
        self.patience = state["patience"]
        self.verbose = state["verbose"]
        self.cooldown = state["cooldown"]
        self.wait = state["wait"]
        self.best = state["best"]
        self.mode = state["mode"]
        backend.set_value(self.model.optimizer.lr, state["lr"])
