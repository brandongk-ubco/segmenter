import tensorflow as tf

class SinActivation(tf.keras.layers.Layer):

    def call(self, x):
        return x - tf.sin(x)

    def compute_output_shape(self, input_shape):
        return input_shape
