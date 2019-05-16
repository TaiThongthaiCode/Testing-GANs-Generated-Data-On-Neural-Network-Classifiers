"""
Creates the npy array by combining 10 generated images.
-> X_train_cifar10.npy and y_train_cifar10.npy
"""

from __future__ import print_function
import os
from collections import defaultdict

try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image
from six.moves import range
import keras.backend as K
from keras.datasets import cifar10
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal
from keras.utils.generic_utils import Progbar
from Minibatch import MinibatchDiscrimination
import matplotlib.pyplot as plt
from keras.layers.noise import GaussianNoise
import numpy as np



def vis_square(data, padsize=1, padval=0):

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data

def main():

    #creates the x_train dataset
    image = np.load("cifar_image_data_952.npy")

    for i in range(9):
        string = "cifar_image_data_" +str(953 +i) + ".npy"
        image_i = np.load(string)
        image = np.concatenate([image, image_i], axis = 0)
    np.save("X_test_cifar10", image)

    y_label = []
    y = 0
    for i in range(1000):
        if i%10 == 0:
            y += 1
        if i%100 == 0:
            y = 0
        y_label.append(y)
    np.save("y_test_cifar10", y_label)

    #save image
    print(image.shape)
    img = vis_square(image)
    Image.fromarray(img).save("test_image.png")



main()
