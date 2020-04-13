# https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, SpatialDropout2D, UpSampling2D, Input, concatenate, LeakyReLU
from tensorflow.keras import backend as K
import tensorflow as tf

def norm_and_activation(c, norm, activation=None):
    if norm == 'batch':
        c = BatchNormalization()(c)
    if norm == 'layer':
        c = LayerNormalization()(c)
    if activation is not None:
        c = activation()(c)
    return c

def upsample_conv(
    inputs,
    filters,
    kernel_size,
    strides,
    padding,
    dropout=0.,
    norm='batch',
    activation=LeakyReLU,
    kernel_initializer='zeros',
    max_dropout=0.5):

    dropout = min(dropout, max_dropout)
    dropout = max(dropout, 0.)

    c = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, activation='linear', kernel_initializer=kernel_initializer)(inputs)
    c = norm_and_activation(c, norm, activation)
    if dropout > 0.0:
        c = SpatialDropout2D(dropout)(c)
    return c


def upsample_simple(filters, kernel_size, strides, padding):
    return UpSampling2D(strides)

def conv2d_block(
    inputs, 
    norm='batch', 
    dropout=0., 
    filters=16, 
    kernel_size=(3,3), 
    kernel_initializer='zeros', 
    padding='same',
    activation=LeakyReLU,
    max_dropout=0.5):
    
    dropout = min(dropout, max_dropout)
    dropout = max(dropout, 0.)

    c = Conv2D(filters, kernel_size, activation='linear', kernel_initializer=kernel_initializer, padding=padding) (inputs)
    c = norm_and_activation(c, norm, activation)
    if dropout > 0.0:
        c = SpatialDropout2D(dropout)(c)

    c = Conv2D(filters, kernel_size, activation='linear', kernel_initializer=kernel_initializer, padding=padding) (c)
    c = norm_and_activation(c, norm, activation)
    if dropout > 0.0:
        c = SpatialDropout2D(dropout)(c)
    return c

def custom_unet(
    input_shape,
    num_classes=1,
    norm='batch', 
    upsample_mode='deconv', # 'deconv' or 'simple' 
    use_dropout_on_upsampling=False, 
    dropout=0.0,
    dropout_change_per_layer=0.0,
    filters=16,
    num_layers=4,
    activation=LeakyReLU,
    output_activation='sigmoid', # 'sigmoid' or 'softmax'
    kernel_initializer='zeros',
    max_dropout=0.5,
    filter_ratio=2):
    
    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(shape=input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, norm=norm, dropout=dropout, max_dropout=max_dropout, activation=activation, kernel_initializer=kernel_initializer)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        dropout += dropout_change_per_layer
        filters = max(int(filters*filter_ratio), 1)

    x = conv2d_block(inputs=x, filters=filters, norm=norm, dropout=dropout, max_dropout=max_dropout, activation=activation, kernel_initializer=kernel_initializer)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):
        filters = max(int(filters // filter_ratio), 1)
        dropout -= dropout_change_per_layer
        x = upsample(inputs=x, filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', norm=norm, dropout=dropout, max_dropout=max_dropout, activation=activation, kernel_initializer=kernel_initializer)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, norm=norm, dropout=dropout, max_dropout=max_dropout, activation=activation, kernel_initializer=kernel_initializer)
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
