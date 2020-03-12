import tensorflow as tf
from tensorflow.keras.layers import Average

@tf.custom_gradient
def AverageSingleGradientOp(inputs):

    def custom_grad(dy):
        weighted = tf.concat(
            [tf.zeros_like(inputs[:-1, :, :, :]),  tf.expand_dims(tf.ones_like(inputs[-1, :, :, :]), 0) / inputs.shape[0]],
            0
        )
        return dy * weighted
    
    return tf.reduce_mean(inputs, 0), custom_grad

class AverageSingleGradient(Average):
        
    def _merge_function(self, inputs):
        return AverageSingleGradientOp(inputs)
