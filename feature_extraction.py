#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Extracao de caracteristicas usando a AlexNet pre-treinada na ImageNet

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
from alexnet import *

K.set_image_dim_ordering('th')

#Train and test of ConvNet
def convnet():
    model = AlexNet('alexnet_weights.h5')
    epochs = 2
    lrate = 0.1
    decay = lrate/epochs
    sgd = SGD(lr=lrate, decay=decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

#Load data from Food-11 dataset
dataset = np.load('./data/trab2_dataset.npz')
imgs = dataset['data']
if K.image_data_format() == 'channels_first':
    imgs = imgs.transpose(0, 3, 1, 2)

#Cria a CNN
alexnet_model = convnet()

#Extrai dados da primeira camada convolucional
modelC1 = Model(inputs=alexnet_model.input, outputs=alexnet_model.get_layer('convpool_1').output)
print "Obtendo features para conv_1"
c1_features = modelC1.predict(imgs)
print "Features obtidas em conv_1: {0}".format(c1_features.shape)
print "Salvando em arquivo npy...\n"
np.savez_compressed('./data/trab2_conv1', data=c1_features)
del c1_features
gc.collect()

#Dados da camada convolucional 5
modelC2 = Model(inputs=alexnet_model.input, outputs=alexnet_model.get_layer('convpool_5').output)
print "Obtendo features para conv_5"
c5_features = modelC2.predict(imgs)
print "Features obtidas em conv_5: {0}".format(c5_features.shape)
print "Salvando em arquivo npy...\n"
np.savez_compressed('./data/trab2_conv5', data=c5_features)
del c5_features
gc.collect()

#Dados da segunda camada totalmente conectada
modelFC2 = Model(inputs=alexnet_model.input, outputs=alexnet_model.get_layer('dense_2').output)
print "Obtendo features para dense_2"
fc2_features = modelFC2.predict(imgs)
print "Features obtidas em dense_2: {0}".format(fc2_features.shape)
print "Salvando em arquivo npy...\n"
np.savez_compressed('./data/trab2_dense2', data=fc2_features)
del fc2_features
gc.collect()

print "Extração de características finalizada !"
