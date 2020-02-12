import tensorflow as tf

class BiasedCosActivation(tf.keras.layers.Layer):

    def call(self, x):
        return x + 0.70710678119 - tf.cos(x + 0.70710678119)

    def compute_output_shape(self, input_shape):
        return input_shape
