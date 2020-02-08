from .CosActivation import CosActivation
from .SincActivation import SincActivation
from tensorflow.keras.layers import LeakyReLU, ReLU, PReLU

def get_activation(activation):
    if activation == "cos":
        return CosActivation
    if activation == "sinc":
        return CosActivation
    if activation == "leaky_relu":
        return LeakyReLU
    if activation == "parametric_relu":
        return PReLU
    if activation == "relu":
        return ReLU
    raise ValueError("Activation %s not defined" % activation)