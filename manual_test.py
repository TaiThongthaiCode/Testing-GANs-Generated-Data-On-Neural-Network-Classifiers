"""
Testing mnist shapes and cifar10 shapes
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.datasets import cifar10


import numpy as np
from PIL import Image
import argparse
import math

def main():

    (X_train_mnist, y_train_mnist), (X_test, y_test) = mnist.load_data()
    (X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()

    print(X_train_mnist.shape)
    print(X_train_cifar.shape)



if __name__ == "__main__":
    main()
