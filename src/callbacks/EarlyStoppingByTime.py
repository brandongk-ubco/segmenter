from tensorflow.keras.callbacks import Callback

import time
import os

class EarlyStoppingByTime(Callback):

    def __init__(self, limit_seconds, verbose=0):
        super(EarlyStoppingByTime, self).__init__()
        self.start = time.time()
        self.limit_seconds = limit_seconds
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = time.time()
        elapsed = current - self.start

        if elapsed > self.limit_seconds:
            print("Epoch %05d: Time limit exhausted (%s seconds)" % (epoch+1, self.limit_seconds))
            time.sleep(60)
            os._exit(123)

        if self.verbose > 0:
            print("Epoch %05d: %.2f seconds remaining" % (epoch+1, self.limit_seconds - elapsed))
