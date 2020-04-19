from tensorflow.keras.layers import Multiply

class LeakyNoisyOr(Multiply):
        
    def _merge_function(self, inputs):
        output = 1. - inputs[0]
        for i in range(1, len(inputs)):
            output *= 1. - inputs[i]
        return 1. - output
