import tensorflow as tf

class ShiftedCosActivation(tf.keras.layers.Layer):

    def call(self, x):
        return x + 1 - tf.cos(x)

    def compute_output_shape(self, input_shape):
        return input_shape
