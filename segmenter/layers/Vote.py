from tensorflow.keras.layers import Average
from tensorflow.keras import backend as K


class Vote(Average):
    def _merge_function(self, inputs):
        output = K.round(inputs[0])
        for i in range(1, len(inputs)):
            output += K.round(inputs[i])
        return output / len(inputs)
