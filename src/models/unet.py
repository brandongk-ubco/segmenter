# https://github.com/karolzak/keras-unet/blob/master/keras_unet/models/custom_unet.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, UpSampling2D, Input, concatenate, Activation, LeakyReLU, ReLU

def upsample_conv(inputs, filters, kernel_size, strides, padding, dropout=0.3, use_batch_norm=True):
    c = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding, activation='linear')(inputs)
    # c = LeakyReLU()(c)
    # if use_batch_norm:
    #     c = BatchNormalization()(c)
    # if dropout > 0.0:
    #     c = Dropout(dropout)(c)
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
    padding='same'):
    
    c = Conv2D(filters, kernel_size, activation='linear', kernel_initializer=kernel_initializer, padding=padding) (inputs)
    c = LeakyReLU()(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)

    c = Conv2D(filters, kernel_size, activation='linear', kernel_initializer=kernel_initializer, padding=padding) (c)
    c = LeakyReLU()(c)
    if use_batch_norm:
        c = BatchNormalization()(c)
    if dropout > 0.0:
        c = Dropout(dropout)(c)
    return c

def custom_unet(
    input_shape,
    num_classes=1,
    use_batch_norm=True, 
    upsample_mode='deconv', # 'deconv' or 'simple' 
    use_dropout_on_upsampling=False, 
    dropout=0.3, 
    dropout_change_per_layer=0.0,
    filters=16,
    num_layers=4,
    output_activation='sigmoid'): # 'sigmoid' or 'softmax'
    
    if upsample_mode=='deconv':
        upsample=upsample_conv
    else:
        upsample=upsample_simple

    # Build U-Net model
    inputs = Input(input_shape)
    x = inputs
    if use_batch_norm:
        x = BatchNormalization()(x)

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
        down_layers.append(x)
        x = MaxPooling2D((2, 2)) (x)
        dropout += dropout_change_per_layer
        filters = filters*2 # double the number of filters with each layer

    x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)

    if not use_dropout_on_upsampling:
        dropout = 0.0
        dropout_change_per_layer = 0.0

    for conv in reversed(down_layers):        
        filters //= 2 # decreasing number of filters with each layer 
        dropout -= dropout_change_per_layer
        x = upsample(inputs=x, filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', use_batch_norm=use_batch_norm, dropout=dropout)
        x = concatenate([x, conv])
        x = conv2d_block(inputs=x, filters=filters, use_batch_norm=use_batch_norm, dropout=dropout)
    
    outputs = Conv2D(num_classes, (1, 1), activation=output_activation) (x)    
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
