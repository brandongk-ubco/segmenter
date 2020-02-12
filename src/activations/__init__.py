from .CosActivation import CosActivation
from .BiasedCosActivation import BiasedCosActivation
from .ShiftedCosActivation import ShiftedCosActivation
from .SincActivation import SincActivation
from .SinActivation import SinActivation
from .ShiftedSinActivation import ShiftedSinActivation
from .CubicActivation import CubicActivation
from tensorflow.keras.layers import LeakyReLU, ReLU, PReLU

def get_activation(activation):
    if activation == "cos":
        return CosActivation
    if activation == "biased_cos":
        return BiasedCosActivation
    if activation == "shifted_cos":
        return ShiftedCosActivation
    if activation == "sin":
        return SinActivation
    if activation == "shifted_sin":
        return ShiftedSinActivation
    if activation == "cubic":
        return CubicActivation
    if activation == "sinc":
        return SincActivation
    if activation == "leaky_relu":
        return LeakyReLU
    if activation == "parametric_relu":
        return PReLU
    if activation == "relu":
        return ReLU
    raise ValueError("Activation %s not defined" % activation)