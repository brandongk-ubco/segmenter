from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
import sys
import os

# NOTE: This does not work.  Looks like you need to actually stop the training and then fit again.
class OptimizerChanger(Callback):

    def __init__(self, stateFile, optimizers, loss, metrics, **kwargs):
        super(OptimizerChanger, self).__init__(**kwargs)
        self.optimizers = optimizers
        self.loss = loss
        self.metrics = metrics
        assert False, "OptimizerChanger currently does not work."

    def on_epoch_begin(self, epoch, logs=None):
        super(OptimizerChanger, self).on_epoch_begin(epoch, logs)

        if not isinstance(self.optimizers, list):
            return

        initial_lr = self.optimizers[0].lr
        lr = K.get_value(self.model.optimizer.lr)

        if initial_lr - lr > sys.float_info.epsilon:
            expected_optimizer = self.optimizers[1]
        else:
            expected_optimizer = self.optimizers[0]

        if not isinstance(self.model.optimizer, expected_optimizer.__class__):
            print("Changing optimizer to {}".format(type(expected_optimizer)))
            self.model.optimizer = expected_optimizer
            self.model.compile(
                optimizer = expected_optimizer,
                loss = self.loss,
                metrics = self.metrics
            )
            K.set_value(self.model.optimizer.lr, lr)
