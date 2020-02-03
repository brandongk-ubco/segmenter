from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam

import os
import pickle

class AdamSaver(Callback):

    def __init__(self, stateFile, **kwargs):
        super(AdamSaver, self).__init__(**kwargs)
        self.stateFile = stateFile

    def on_epoch_end(self, epoch, logs={}):
        super(AdamSaver, self).on_epoch_end(epoch, logs)

        with open(self.stateFile, 'wb') as state_pickle:
            pickle.dump(self.model.optimizer.get_config(), state_pickle)

    def on_train_begin(self, logs=None):
        super(AdamSaver, self).on_train_begin(logs)

        if not os.path.isfile(self.stateFile):
            return

        with open(self.stateFile, 'rb') as state_pickle:
            self.model.optimizer = Adam.from_config(pickle.load(state_pickle))