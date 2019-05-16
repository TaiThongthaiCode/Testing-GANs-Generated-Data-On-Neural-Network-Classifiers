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

import matplotlib.pyplot as plt
import tensorflow as tf

import numpy as np
from PIL import Image
import argparse
import math

def main():

    # (X_train_mnist, y_train_mnist), (X_test, y_test) = mnist.load_data()
    # #(X_train_cifar, y_train_cifar), (X_test_cifar, y_test_cifar) = cifar10.load_data()
    #
    # #print(type(X_train_mnist))
    #
    # X_train_mnist = X_train_mnist[:, :, :, None]
    # s = tf.concat([X_train_mnist, X_train_mnist], axis=0)
    # #print("S SHAPE", s.shape)
    # image = combine_images(X_train_mnist)
    # image = image*127.5+127.5
    # Image.fromarray(image.astype(np.uint8)).save(
    #     "generated_image.png")
    # #print(X_train_cifar.shape)


#     X_train = np.load("fake_mnist_dataset.npy")
#     print(X_train.shape)
#
# def combine_images(generated_images):
#     num = generated_images.shape[0]
#     width = int(math.sqrt(num))
#     height = int(math.ceil(float(num)/width))
#     shape = generated_images.shape[1:3]
#     image = np.zeros((height*shape[0], width*shape[1]),
#                      dtype=generated_images.dtype)
#     for index, img in enumerate(generated_images):
#         i = int(index/width)
#         j = index % width
#         image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
#             img[:, :, 0]
#     return image



    # y_train_raw = np.genfromtxt("y_train_mnist.csv",dtype=int, delimiter=",")
    # y_train = np.zeros(1408, dtype = int)
    # counter = 0
    # for i in range(len(y_train_raw)):
    #     for j in range(len(y_train_raw[i])):
    #
    #         y_train[counter] = y_train_raw[i][j]
    #         if counter == 1407:
    #             break
    #         counter += 1
    #
    # y_train[0] = 1
    # y_train[1407] = 9
    # for i in range(len(y_train)):
    #     print(y_train[i])
    # print("fake:", y_train.shape)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # print(y_train)
    # # for i in range(len(y_train)):
    # #     print(y_train[i])
    # print("real:", y_train.shape)


    plt.imshow(x_train[0] )
    plt.title(" Digit " + str(y_train[0]) )
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # hello = np.load("fake_mnist_dataset.npy")
    # print(hello.shape)
if __name__ == "__main__":
    main()
