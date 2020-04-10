from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, SGD

import os
import pickle

class OptimizerSaver(Callback):

    def __init__(self, stateFile, optimizer_config, loss, metrics, **kwargs):
        super(OptimizerSaver, self).__init__(**kwargs)
        self.stateFile = stateFile
        self.optimizer_config = optimizer_config
        self.restored = False
        self.loss = loss
        self.metrics = metrics

    def on_epoch_end(self, epoch, logs={}):
        super(OptimizerSaver, self).on_epoch_end(epoch, logs)

        with open(self.stateFile, 'wb') as state_pickle:
            pickle.dump(self.model.optimizer.get_config(), state_pickle)

    def on_train_begin(self, logs=None):
        super(OptimizerSaver, self).on_train_begin(logs)

        if not os.path.isfile(self.stateFile):
            return

        if self.restored:
            return

        with open(self.stateFile, 'rb') as state_pickle:
            if isinstance(self.model.optimizer, Adam):
                self.model.optimizer = Adam.from_config(pickle.load(state_pickle))
            elif isinstance(self.model.optimizer, SGD):
                self.model.optimizer = SGD.from_config(pickle.load(state_pickle))
            else:
                raise ValueError("Unknown Optimizer {}".format(optimizer_config["NAME"]))

        self.model.compile(
            optimizer = self.model.optimizer,
            loss = self.loss,
            metrics = self.metrics
        )

        self.restored = True