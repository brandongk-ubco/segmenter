from tensorflow.keras import backend
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import json


class SubModelCheckpoint(ModelCheckpoint):
    def __init__(self, submodel, **kwargs):
        super(SubModelCheckpoint, self).__init__(**kwargs)
        self.submodel = submodel

    def set_model(self, model):
        self.model = model.get_layer(self.submodel)
