# https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, SpatialDropout2D, UpSampling2D, Input, concatenate, LeakyReLU
from tensorflow.keras import backend as K
import tensorflow as tf

def upsample_conv(inputs, filters, kernel_size, strides, padding, dropout=0.3, use_batch_norm=True, activation=LeakyReLU, kernel_initializer='he_normal'):
    c = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, activation='linear', kernel_initializer=kernel_initializer)(inputs)
    if use_batch_norm:
        c = BatchNormalization(scale=True)(c)
    c = activation()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(dropout)(c)
    return c


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def conv2d_block(
    inputs, 
    use_batch_norm=True, 
    dropout=0.3, 
    filters=16, 
    kernel_size=(3,3), 
    kernel_initializer='he_normal', 
    padding='same',
    activation=LeakyReLU):
    
    c = Conv2D(filters, kernel_size, activation='linear', kernel_initializer=kernel_initializer, padding=padding) (inputs)
    if use_batch_norm:
        c = BatchNormalization(scale=True)(c)
    c = activation()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(dropout)(c)

    c = Conv2D(filters, kernel_size, activation='linear', kernel_initializer=kernel_initializer, padding=padding) (c)
    if use_batch_norm:
        c = BatchNormalization(scale=True)(c)
    c = activation()(c)
    if dropout > 0.0:
        c = SpatialDropout2D(dropout)(c)
    return c

def custom_unet(
    input_shape,
    num_classes=1,
    use_batch_norm=True, 
    upsample_mode='deconv', # 'deconv' or 'simple' 
    use_dropout_on_upsampling=False, 
    dropout=0.0,
    dropout_change_per_layer=0.0,
    filters=16,
    num_layers=4,
    activation=LeakyReLU,
    output_activation='sigmoid', # 'sigmoid' or 'softmax'
    kernel_initializer='he_normal'):
    
    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(shape=input_shape)
    x = inputs
    if use_batch_norm:
        x = BatchNormalization()(x)

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, activation=activation)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        dropout += dropout_change_per_layer
        filters = filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, activation=activation)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer 
        dropout -= dropout_change_per_layer
        x = upsample(inputs=x, filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', use_batch_norm=use_batch_norm, dropout=dropout, activation=activation, kernel_initializer=kernel_initializer)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout, activation=activation)
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
