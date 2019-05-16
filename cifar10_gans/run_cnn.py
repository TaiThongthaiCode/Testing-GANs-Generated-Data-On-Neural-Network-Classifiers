"""
Starter code for NN training and testing.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Sara Mathieson, Richard Muniu, Sam Yan
Date: 07 April, 2019
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch


import cnn as cn

##################

# Globals for the device
USE_GPU = True

if USE_GPU:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'


print('Using device: ', device)

##################

class Dataset(object):

    def __init__(self, X, y, batch_size, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        Inputs:
        - X: Numpy array of data, of any shape
        - y: Numpy array of labels, of any shape with y.shape[0] == X.shape[0]
        - batch_size: Integer giving number of elements per minibatch
        - shuffle: (optional) Boolean, whether to shuffle the data on each epoch
        """
        assert X.shape[0] == y.shape[0], 'Different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        """
        Implementing __iter__ allows us to use enumerate(dataset) to iterate
        through our data.
        """
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.y[i:i+B]) for i in range(0, N, B))

def combine_batches(path):
    """
    Path points to the directory cifar-10-batches-py. Code based on:
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/keras/
        datasets/cifar10.py
    Reads in all 5 batches, combines them, then separates into train and test.
    """
    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def load_cifar10(path, num_training=900, num_validation=100, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for a neural net classifier.
    """
    # load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = combine_batches(path)
    (X_train, y_train), (X_test, y_test) = cifar10


    X_train = np.load("X_train_cifar10.npy")
    y_train = np.load("y_train_cifar10.npy")
    X_val = X_train
    y_val = y_train
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()


    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # normalize the data. First find the mean and std of the *training* data,
    # then subtract off this mean from each dataset and divide by the std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel)/std_pixel
    X_val = (X_val - mean_pixel)/std_pixel
    X_test = (X_test - mean_pixel)/std_pixel
    ######## END YOUR CODE #############

    return X_train, y_train, X_val, y_val, X_test, y_test

def training_step(scores, y, params, learning_rate):
    """
    Set up the part of the computational graph which makes a training step.

    Inputs:
    - scores: TensorFlow Tensor of shape (N, C) giving classification scores for
      the model.
    - y: TensorFlow Tensor of shape (N,) giving ground-truth labels for scores;
      y[i] == c means that c is the correct class for scores[i].
    - params: List of TensorFlow Tensors giving the weights of the model
    - learning_rate: Python scalar giving the learning rate to use for gradient
      descent step.

    Returns:
    - loss: A TensorFlow Tensor of shape () (scalar) giving the loss for this
      batch of data; evaluating the loss also performs a gradient descent step
      on params.
    """
    # First compute the loss; the first line gives losses for each example in
    # the minibatch, and the second averages the losses acros the batch
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, \
        logits=scores)
    loss = tf.reduce_mean(losses)

    # Compute the gradient of the loss with respect to each parameter of the the
    # network. This is a very magical function call: TensorFlow internally
    # traverses the computational graph starting at loss backward to each
    # element of params, and uses backpropagation to figure out how to compute
    # gradients; it then adds new operations to the computational graph which
    # compute the requested gradients, and returns a list of TensorFlow Tensors
    # that will contain the requested gradients when evaluated.
    grad_params = tf.gradients(loss, params)

    # Make a gradient descent step on all of the model parameters.
    new_weights = []
    for w, grad_w in zip(params, grad_params):
        new_w = tf.assign_sub(w, learning_rate * grad_w)
        new_weights.append(new_w)

    # Insert a control dependency so that evaluting the loss causes a weight
    # update to happen; then return the loss
    with tf.control_dependencies(new_weights):
        return tf.identity(loss)

def train(train_dset, val_dset, model_fn, init_fn, learning_rate,test_dset):
    """
    Train a model on CIFAR-10.
    Inputs:
    - train_dset & val_dset (training and validation data, respectively)
    - model_fn: A Python function that performs the forward pass of the model
      using TensorFlow; it should have the following signature:
      scores = model_fn(x, params) where x is a TensorFlow Tensor giving a
      minibatch of image data, params is a list of TensorFlow Tensors holding
      the model weights, and scores is a TensorFlow Tensor of shape (N, C)
      giving scores for all elements of x.
    - init_fn: A Python function that initializes the parameters of the model.
      It should have the signature params = init_fn() where params is a list
      of TensorFlow Tensors holding the (randomly initialized) weights of the
      model.
    - learning_rate: Python float giving the learning rate to use for SGD.
    """
    # First clear the default graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')

    # Set up the computational graph for performing forward and backward passes,
    # and weight updates.
    with tf.device(device):
        ###### TODO 6: YOUR CODE HERE ######
        # Set up placeholders for the data and labels

        x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        y = tf.placeholder(tf.int32, shape=(None))
        params = init_fn()              # Initialize the model parameters
        scores = model_fn(x, params)    # Forward pass of the model
        loss = training_step(scores, y, params, learning_rate) # Loss after one training step


        ######## END YOUR CODE #############
    accuracies = []
    losses = []
    # Now we actually run the graph many times using the training data


    with tf.Session() as sess:
        # Initialize variables that will live in the graph
        sess.run(tf.global_variables_initializer())
        for e in range(1, 100):
            for t, (x_np, y_np) in enumerate(train_dset):
                # Run the graph on a batch of training data; recall that asking
                # TensorFlow to evaluate loss will cause an SGD step to happen.
                feed_dict = {x: x_np, y: y_np}
                loss_np = sess.run(loss, feed_dict=feed_dict)
                # Periodically print the loss and check accuracy on the val set
                #accuracy = 0.0 #init

                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss_np))
                accuracy = check_accuracy(sess, val_dset, x, scores, is_training, verbose=False,test=False)
                accuracies.append(accuracy)
                losses.append(loss_np)

        print("TESTING: ")
        test_predictions,test_labels = check_accuracy(sess, test_dset, x, scores, is_training, verbose=True,test=True)

    #print(test_predictions)
    return accuracies, losses, test_predictions,test_labels


def print_confusion_matrix(test_labels, predictions):
    """
    Prints a confusion matrix
    """
    con_mat = tf.confusion_matrix(test_labels, predictions)
    with tf.Session():
        print(tf.Tensor.eval(con_mat,feed_dict=None, session=None)) #from stackoverflow

def check_accuracy(sess, dset, x, scores, is_training=None, verbose=False,test=False):
    """
    Check accuracy on a classification model.
    Inputs:
    - sess: A TensorFlow Session that will be used to run the graph
    - dset: A Dataset object on which to check accuracy
    - x: A TensorFlow placeholder Tensor where input images should be fed
    - scores: A TensorFlow Tensor representing the scores output from the
      model; this is the Tensor we will ask TensorFlow to evaluate.
    - verbose: A boolean; if true this method will print out the accuracy post-evaluation
    """
    num_correct, num_samples = 0, 0
    preds = []
    labels = []
    for x_batch, y_batch in dset:
        feed_dict = {x: x_batch, is_training: 0}
        scores_np = sess.run(scores, feed_dict=feed_dict)
        y_pred = scores_np.argmax(axis=1)
        num_samples += x_batch.shape[0]
        num_correct += (y_pred == y_batch).sum()
        preds.extend(y_pred)
        labels.extend(y_batch)
    acc = float(num_correct) / num_samples
    if verbose:
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100*acc))
    if test:
        return preds,labels
    return acc

def main():
    # Invoke the above function to get our data
    path = "/home/smathieson/public/cs66/cifar-10-batches-py/"
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(path)
    print('Train data shape: ', X_train.shape)              # (49000, 32, 32, 3)
    print('Train labels shape: ', y_train.shape)            # (49000,)
    print('Validation data shape: ', X_val.shape)           # (1000, 32, 32, 3)
    print('Validation labels shape: ', y_val.shape)         # (1000,)
    print('Test data shape: ', X_test.shape)                # (10000, 32, 32, 3)
    print('Test labels shape: ', y_test.shape)              # (10000,)

    print(y_test)
    ###### TODO 2: YOUR CODE HERE ######
    # set up train_dset, val_dset, and test_dset, all as Dataset objects
    # train should be shuffled, but not validation and testing datasets
    train_dset = Dataset(X_train, y_train, 64, shuffle=True)
    val_dset = Dataset(X_val, y_val, 64)
    test_dset = Dataset(X_test, y_test, 64)
    ######## END YOUR CODE #############

    # test iterating through this dataset (TODO uncomment this!)
    for t, (x, y) in enumerate(train_dset):
        print(t, x.shape, y.shape)
        if t > 5: break

    # call the train function to train a three-layer CNN
    cnn_acc, cnn_loss,cnn_preds,cnn_labels = train(train_dset, val_dset, cn.three_layer_convnet, cn.three_layer_convnet_init, 3e-3,test_dset)

    # Create confusion matrices for both methods using the test data
    print("Convolutional Confusion Matrix:")
    print_confusion_matrix(cnn_labels, cnn_preds)


if __name__ == "__main__":
    main()
