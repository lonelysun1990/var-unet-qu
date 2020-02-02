##################################################################
# Implemented based on v-unet by Esser.
# with necessary simplification for reservoir simulation problems.
# Larry Jin
# May, 2019
# Stanford University
##################################################################

from layers import *

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, concatenate
from keras.models import Model

import numpy as np
import copy


def create_enc_up(input_shape, n_scales = 1, n_filters = 8, n_res_blk = 1, max_filters = 128):
    '''
    Following the naming convension of v-unet (Esser).
    Get mean of posterior p(z|x, y_hat) from input x, y_hat
    
    @params: input_shape
    @output: model for enc_up

    enc_up:
    @params: x: the image input, perm field (or, should it be saturation field) 
             c: shape, well location/control
    @output: z: I don't know what it is yet (or, should it be perm field)
             z is just a mean of the posterior
    '''
    input = Input(shape=input_shape, name='image')

    # outputs
    hs = []
    hidden_shapes = []
    h = Conv2D(n_filters, (1,1), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(input)
    for l in range(n_scales):
        for i in range(n_res_blk):
            h = res_conv(n_filters, 3, 3)(h)
            hs.append(h)
            # print(h.shape.as_list()[1:])
            hidden_shapes.append(tuple(h.shape.as_list()[1:]))
        if l + 1 < n_scales:
            n_filters = min(n_filters*2, max_filters)
            h = conv_bn_relu(n_filters, 3, 3, (2, 2))(h) # down sample
    output = hs

    # return both the model and the hidden_shapes
    return [Model(input, output, name='enc_up'), hidden_shapes]


def create_enc_down(latent_shapes):
    '''
    Following the naming convension of v-unet (Esser).
    Input mean of posterior p(z|x, y_hat)
    Sample the posterior


    dec_down:
    @param:
            z : mean of the posterior

    @output:
            z_posterior_sample: sampled posterior
    '''
    
    input = Input(shape=latent_shapes, name='z_mean')

    z_posterior_mean = input

    sampler = create_sampler(t_sigma = 1.0)
    z_posterior_sample = sampler(input)

    output = [z_posterior_sample, z_posterior_mean]
    return Model(input, output, name='enc_down')


def create_dec_up(input_shape, n_scales = 1, n_filters = 8, n_res_blk = 1, max_filters = 128):
    '''
    Following the naming convension of v-unet (Esser).
    Basically the same as enc_up, input only has one channel: y_hat

    dec_up:
    @params: 
             c: shape, well location/control
    @output: 
             z: mean of the piror p(z|y_hat)
    '''
    input = Input(shape=input_shape, name='image')

    # outputs
    hs = []
    h = Conv2D(n_filters, (1,1), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(input)
    for l in range(n_scales):
        for i in range(n_res_blk):
            h = res_conv(n_filters, 3, 3)(h)
            hs.append(h)
        if l + 1 < n_scales:
            n_filters = min(n_filters*2, max_filters)
            h = conv_bn_relu(n_filters, 3, 3, (2, 2))(h) # down sample
    output = hs

    return Model(input, output, name='dec_up')

def create_dec_down(hidden_shapes, latent_shape, n_scales = 1, n_res_blk = 1):
    '''
    Following the naming convension of v-unet (Esser).
    Take z_posterior from enc_up (x,c)
    Take gs from dec_up (c)
    Sample the piror
    
    dec_down:
    @params:
            z_posterior: sampled z posterior (or, just mean for now)
            gs: hidden activation from skip connection (output from dec_up)

    @output:
            x_hat: reconstructed x (saturation)

    '''
    hs = []
    gs = []
    gs_= [] # backup for gs_
    for s in hidden_shapes:
        g_i = Input(shape=s)
        gs.append(g_i)
        gs_.append(g_i)
        
    z_posterior = Input(shape=latent_shape) # input for the model

    sampler = create_sampler(t_sigma = 1.0)
    z_piror_mean = gs[-1]
    z_piror_sample = sampler(z_piror_mean) #

    n_filters = z_posterior.shape.as_list()[-1] # say shape is (8, 8, 128), n_filters is 128

    h = Conv2D(n_filters, (1,1), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(z_posterior)
    # print(h.shape.as_list())
    for l in range(n_scales):
        for i in range(n_res_blk):
            h = concatenate([h, gs.pop()], axis=3)
            h = res_conv(n_filters*2, 3, 3)(h)
            h = Conv2D(n_filters, (3,3), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(h)
            # print(h.shape.as_list())
            hs.append(h)

        if l + 1 < n_scales:
            n_filters = gs[-1].shape.as_list()[-1]
            h = dconv_bn_nolinear(n_filters, 3, 3, stride=(2, 2))(h) # up sample

    x_hat = Conv2D(1, (3,3), strides=(1,1), padding='same', kernel_regularizer=regularizers.l2(reg_weights))(h)

    return Model(inputs = [z_posterior, *gs_], outputs = [x_hat, z_piror_sample, z_piror_mean], name='dec_down')

def sample(t_mean, t_sigma):
    '''
    Draws samples from a standard normal and scales the samples with
    standard deviation of the variational distribution and shifts them
    by the mean.

    @params:
        args: sufficient statistics of the variational distribution.

    @output:
        Samples from the variational distribution.
    '''
    epsilon = K.random_normal(shape=K.shape(t_mean), mean=0., stddev=1.)
    return t_mean + t_sigma * epsilon


def create_sampler(t_sigma = 1.0):
    '''
    Creates a sampling layer.
    '''
    return Lambda(lambda x: sample(x, t_sigma), name='sampler')
