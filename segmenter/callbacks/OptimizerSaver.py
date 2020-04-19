from tensorflow.keras.callbacks import Callback

import os
import pickle


class OptimizerSaver(Callback):
    def __init__(self, stateFile, loss, metrics, **kwargs):
        super(OptimizerSaver, self).__init__(**kwargs)
        self.stateFile = stateFile
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
            self.model.optimizer = self.model.optimizer.from_config(
                pickle.load(state_pickle))

        self.model.compile(optimizer=self.model.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        self.restored = True