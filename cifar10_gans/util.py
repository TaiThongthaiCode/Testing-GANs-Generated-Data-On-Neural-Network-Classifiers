"""
Starter code for utility functions.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Sara Mathieson, Richard Muniu, Sam Yan
Date: 07 April, 2019
"""

import numpy as np
import tensorflow as tf

def flatten(x):
    """
    Input:
    - TensorFlow Tensor of shape (n, D1, ..., DM)

    Output:
    - TensorFlow Tensor of shape (n, D1 * ... * DM)
    """
    ###### TODO 3: YOUR CODE HERE ######
    # flatten the data by unraveling all dimensions besides the first one
    # Hint: look up tf.shape and tf.reshape
    ######## END YOUR CODE #############
    return tf.reshape(x, [tf.shape(x)[0], -1])

def kaiming_normal(shape):
    """
    Normalization method that accounts for the number of inputs into each node.
    He et al, "Delving Deep into Rectifiers: Surpassing Human-Level Performance
    on ImageNet Classification"
    """
    if len(shape) == 2:
        fan_in = shape[0]
    elif len(shape) == 4:
        fan_in = np.prod(shape[:3])
    return tf.random_normal(shape) * np.sqrt(2.0 / fan_in)
