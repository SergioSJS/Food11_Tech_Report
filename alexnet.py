#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
AlexNet pre-treinada na ImageNet

Frameworks: Theano 0.9 + Keras 2.0
'''
import os, numpy as np, gc

#Keras imports
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Lambda
from keras import backend as K
from keras.constraints import maxnorm
from keras.optimizers import SGD

K.set_image_dim_ordering('th')

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        K.set_image_dim_ordering('th')
        if K.backend()=='tensorflow':
            b, ch, r, c = X.get_shape()
        else:
            b, ch, r, c = X.shape            
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1)), padding=((0,0),(half,half)) )
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            if K.backend()=='tensorflow':
                ch = int(ch)
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)

def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    def f(X):
        if K.backend()=='tensorflow':
            div = int(X.get_shape()[axis]) // ratio_split
        else:
            div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)

#Alex-net architeture
def AlexNet(weights_path=None):
    inputs = Input(shape=(3,227,227))
    conv_1 = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(inputs)
    conv_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = Concatenate(axis=1, name='conv_2')([
                       Conv2D(128, (5, 5), activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)])
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Conv2D(384, (3, 3), activation='relu', name='conv_3')(conv_3)
    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = Concatenate(axis=1, name='conv_4')([
                       Conv2D(192, (3, 3), activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)])
    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = Concatenate(axis=1, name='conv_5')([
                       Conv2D(128, (3, 3), activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)])
    dense_1 = MaxPooling2D((3, 3), strides=(2, 2), name='convpool_5')(conv_5)
    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)
    model = Model(inputs=inputs, outputs=prediction)

    if weights_path:
        model.load_weights(weights_path)

    return model
