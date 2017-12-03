#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Script para montar o dataset utilizado no trabalho.

Mantém as imagens no formato RGB, com ordens na matriz: largura, altura e canais
'''
from PIL import Image
import os, numpy as np, glob
folder = './data/training'
max_size = 100

#Carrega imagens no tamanho necessario pela AlexNet
def load_image(path, targetSize=(227,227)):
    im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert("RGB")
    if im.size != targetSize:
        im = im.resize(targetSize)
    return np.asarray(im)

#Percorre a lista de imagens jpg
ims = []
labels = []
d = {x:0 for x in xrange(11)}
for f in glob.glob(os.path.join(folder, '*.jpg')):
    label = int(os.path.basename(f).split('_')[0])
    if d[label] == max_size:
        continue
    d[label] += 1
    labels.append(label)
    ims.append(load_image(f))
    
#Gera os arrays numpy
im_array = np.array(ims, dtype='uint8')
lb_array = np.array(labels, dtype='uint8')

#Salva o dataset
np.savez_compressed('./data/trab2_dataset', data=im_array, labels=lb_array)

