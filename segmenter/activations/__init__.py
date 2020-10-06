from tensorflow.keras.layers import LeakyReLU, ReLU, PReLU, Activation, ELU
import tensorflow as tf


def sinc_activation(x):
    x = tf.where(tf.abs(x) < 1e-20, 1e-20 * tf.ones_like(x), x)
    return x + tf.sin(x) / x


def get_activation(activation):
    if activation == "cos":
        return lambda: Activation(lambda x: x - tf.cos(x))
    if activation == "biased_cos":
        return lambda: Activation(lambda x: x + 0.70710678119 - tf.cos(
            x + 0.70710678119))
    if activation == "shifted_cos":
        return lambda: Activation(lambda x: x + 1 - tf.cos(x))
    if activation == "sin":
        return lambda: Activation(lambda x: x - tf.sin(x))
    if activation == "shifted_sin":
        return lambda: Activation(lambda x: x - tf.sin(x) - 1)
    if activation == "cubic":
        return lambda: Activation(lambda x: tf.where(
            tf.abs(x) > 1, x, tf.math.pow(x, 3)))
    if activation == "sinc":
        return lambda: Activation(lambda x: sinc_activation(x))
    if activation == "leaky_relu":
        return LeakyReLU
    if activation == "parametric_relu":
        return PReLU
    if activation == "relu":
        return ReLU
    if activation == "elu":
        return ELU
    if activation == "linear":
        return lambda: Activation(lambda x: x)
    if activation == "swish":
        return lambda: Activation(lambda x: x * tf.keras.activations.sigmoid(x)
                                  )
    raise ValueError("Activation %s not defined" % activation)
