from tensorflow.keras import backend as K


def logit(x):
    """ Computes the logit function, i.e. the logistic sigmoid inverse. """
    x = K.clip(x, K.epsilon(), 1 - K.epsilon())
    return - K.log(1. / x - 1.)
