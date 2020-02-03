import tensorflow as tf

class CosActivation(tf.keras.layers.Layer):

    def call(self, x):
        return x - tf.cos(x)

    def compute_output_shape(self, input_shape):
        return input_shape
