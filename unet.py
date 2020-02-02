from layers import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate
from keras.models import Model

import numpy as np


def create_unet_up(input_shape, n_scales = 3, n_filters = 16, max_filters = 128):
    # Encoder
    input = Input(shape=input_shape, name='image')

    hs = []
    hidden_shapes = []
    h = input
    for l in range(n_scales):
        h =  conv_bn_relu(n_filters, 3, 3, stride=(2,2))(h)
        n_filters = min(n_filters*2, max_filters)
        hs.append(h)
        hidden_shapes.append(tuple(h.shape.as_list()[1:]))

        h = conv_bn_relu(n_filters, 3, 3, stride=(1,1))(h)
        n_filters = min(n_filters*2, max_filters)
        hs.append(h)
        hidden_shapes.append(tuple(h.shape.as_list()[1:]))

    output = hs

    return [Model(input, output, name='unet_up'), hidden_shapes]


def create_unet_down(hidden_shapes, n_scales = 3, n_res_blk = 6):

    gs = []
    for s in hidden_shapes:
        gs.append(Input(shape=s))
    input = gs

    h = gs[-1]
    for _ in range(n_res_blk):
        h = res_conv(128, 3, 3)(h)

    for l in range(n_scales):
        # print("l=",l)
        # print("concat:   gs[-l*2-1]=",gs[-l*2-1].shape.as_list())
        # print("n_filter: gs[-l*2-2]=",gs[-l*2-2].shape.as_list())
        # print("current shape: h=", h.shape.as_list())

        n_filters = gs[-l*2-2].shape.as_list()[-1]
        h = concatenate([gs[-l*2-1], h], axis=-1)
        h = dconv_bn_nolinear(n_filters, 3, 3, stride=(1, 1))(h)

        # print("l=",l)
        # print("concat:   gs[-l*2-2]=",gs[-l*2-2].shape.as_list())
        if l+1 < n_scales:
            n_filters = gs[-l*2-3].shape.as_list()[-1]
            # print("n_filter: gs[-l*2-3]=",gs[-l*2-3].shape.as_list())

        else:
            n_filters = 2
            # print("n_filter=2")        
        
        # print("current shape: h=", h.shape.as_list())
        h = concatenate([gs[-l*2-2], h], axis=-1)
        h = dconv_bn_nolinear(n_filters, 3, 3, stride=(2, 2))(h)

    output = Conv2D(1, (3, 3), padding='same', activation=None)(h)
    # print("output shape: ", output.shape.as_list())

    return Model(input, output)