import tensorflow as tf

class ShiftedSinActivation(tf.keras.layers.Layer):

    def call(self, x):
        return x - tf.sin(x) - 1

    def compute_output_shape(self, input_shape):
        return input_shape
