from layers import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate
from keras.models import Model

import numpy as np


def create_vae(input_shape):
    # Encoder
    input = Input(shape=input_shape, name='image')

    enc1_conv = Conv2D(16, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.00001))(input)
    enc1_bn_relu = bn_relu()(enc1_conv)
    
    enc2_conv = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc1_bn_relu)
    enc2_bn_relu = bn_relu()(enc2_conv)
    
    enc3_conv = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc2_bn_relu)
    enc3_bn_relu = bn_relu()(enc3_conv)
    
    enc4_conv = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc3_bn_relu)
    enc4_bn_relu = bn_relu()(enc4_conv)
    
    enc5_conv = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc4_bn_relu)
    enc5_bn_relu = bn_relu()(enc5_conv)
    
    enc6_conv = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(0.00001))(enc5_bn_relu)
    enc6_bn_relu = bn_relu()(enc6_conv)


    x0 = res_conv(128, 3, 3)(enc6_bn_relu)
    x1 = res_conv(128, 3, 3)(x0)
    x2 = res_conv(128, 3, 3)(x1)
    x3 = res_conv(128, 3, 3)(x2)
    x4 = res_conv(128, 3, 3)(x3)
    
    dec6 = res_conv(128, 3, 3)(x4)

    merge6 = concatenate([enc6_bn_relu, dec6], axis=3)
    dec5 = dconv_bn_nolinear(128, 3, 3, stride=(1, 1))(merge6)
    merge5 = concatenate([enc5_bn_relu, dec5], axis=3)
    dec4 = dconv_bn_nolinear(128, 3, 3, stride=(2, 2))(merge5)
    merge4 = concatenate([enc4_bn_relu, dec4], axis=3)
    dec3 = dconv_bn_nolinear(64, 3, 3, stride=(1, 1))(merge4)
    merge3 = concatenate([enc3_bn_relu, dec3], axis=3)
    dec2 = dconv_bn_nolinear(64, 3, 3, stride=(2, 2))(merge3)
    merge2 = concatenate([enc2_bn_relu, dec2], axis=3)
    dec1 = dconv_bn_nolinear(32, 3, 3, stride=(1, 1))(merge2)
    merge1 = concatenate([enc1_bn_relu, dec1], axis=3)
    dec0 = dconv_bn_nolinear(16, 3, 3, stride=(2, 2))(merge1)

    final_out = Conv2D(1, (3, 3), padding='same', activation=None)(dec0)
    
    output = final_out

    # Full net
    vae_model = Model(input, output)

    return vae_model