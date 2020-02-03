import tensorflow as tf

class SincActivation(tf.keras.layers.Layer):

    def call(self, x):
        x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
        return x + tf.sin(x) / x

    def compute_output_shape(self, input_shape):
        return input_shape
