"""
Starter code for convolutional neural network.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Sara Mathieson, Richard Muniu, Sam Yan
Date: 07 April, 2019
"""

import tensorflow as tf
import numpy as np

import util

def three_layer_convnet(x, params):
    """
    A three-layer convolutional network with the architecture described above.
    Inputs:
    - x: A tensor of shape (N, 32, 32, 3) giving a minibatch of images
    - params: A list of TensorFlow Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: TensorFlow Tensor of shape (KH1, KW1, 3, channel_1) giving
        weights for the first convolutional layer.
      - conv_b1: TensorFlow Tensor of shape (channel_1,) giving biases for the
        first convolutional layer.
      - conv_w2: TensorFlow Tensor of shape (KH2, KW2, channel_1, channel_2)
        giving weights for the second convolutional layer
      - conv_b2: TensorFlow Tensor of shape (channel_2,) giving biases for the
        second convolutional layer.
      - fc_w: TensorFlow Tensor giving weights for the fully-connected layer.
        Can you figure out what the shape should be?
      - fc_b: TensorFlow Tensor giving biases for the fully-connected layer.
        Can you figure out what the shape should be?
    """
    ###### TODO 8: YOUR CODE HERE ######
    # set up a 3-layer CNN (following the lab writeup)
    strides = [1,1,1,1]
    padding = 'SAME'

    conv_stage_1 = tf.nn.conv2d(x, params[0] ,strides, padding) + params[1]
    conv1_output = tf.nn.relu(conv_stage_1)
    conv1_output = tf.nn.max_pool(conv1_output,[1,2,2,1],[1,1,1,1],'SAME')

    conv_stage_2 = tf.nn.conv2d(conv1_output, params[2], strides, padding) + params[3]
    conv2_output = tf.nn.relu(conv_stage_2)
    conv2_output = tf.nn.max_pool(conv2_output,[1,2,2,1],[1,1,1,1],'SAME')

    """
    conv_stage_3 = tf.nn.conv2d(conv2_output, params[4], strides, padding) + params[5]
    conv3_output = tf.nn.relu(conv_stage_3)
    """
    fc_weights = util.flatten(conv2_output)
    scores = tf.matmul(fc_weights, params[4]) + params[5]
    return scores



def three_layer_convnet_init():
    """
    Initialize the weights of a Three-Layer ConvNet, for use with the
    three_layer_convnet function defined above.
    Returns a list containing:
    - conv_w1: TensorFlow Variable giving weights for the first conv layer
    - conv_b1: TensorFlow Variable giving biases for the first conv layer
    - conv_w2: TensorFlow Variable giving weights for the second conv layer
    - conv_b2: TensorFlow Variable giving biases for the second conv layer
    - fc_w: TensorFlow Variable giving weights for the fully-connected layer
    - fc_b: TensorFlow Variable giving biases for the fully-connected layer
    """
    ###### TODO 9: YOUR CODE HERE ######
    # initialize the weights of the 3-layer CNN, using filter sizes:
    # first layer: 32 filters, each with size 5x5
    conv_w1 = tf.Variable(util.kaiming_normal([5, 5, 3, 32]))
    conv_b1 = tf.Variable(tf.zeros((32,)))
    # second layer: 16 filters, each with size 3x3
    conv_w2 = tf.Variable(util.kaiming_normal([3, 3, 32, 16]))
    conv_b2 = tf.Variable(tf.zeros(16,))
    # third layer: fully-connected layer to compute scores for 10 classes

    fc_w = tf.Variable(util.kaiming_normal([32 * 32 * 16, 10])) # final network to outputs
    fc_b = tf.Variable(tf.zeros((10),))

    params = [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b]
    ######## END YOUR CODE #############
    return params

def main():

    device = '/cpu:0'
    print('Using device: ', device)

    # test 3-layer network (should get shape (64,10))
    three_layer_convnet_test(device, np.zeros((64, 32, 32, 3)))

if __name__ == "__main__":
    main()
