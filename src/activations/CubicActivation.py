import tensorflow as tf

class CubicActivation(tf.keras.layers.Layer):

    def call(self, x):
        return tf.where(tf.abs(x) > 1, x, tf.math.pow(x,3))

    def compute_output_shape(self, input_shape):
        return input_shape
