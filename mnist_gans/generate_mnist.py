"""
Program to generate fake mnist data set.
Requires training in run_gans.py before generation becomes sufficiently valid
"""

from run_gans import *

import tensorflow as tf

import numpy as np
from PIL import Image
import argparse
import math

def get_args():
    """
    Gets command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

def generate_data_set():
    """
    Method to generate and save dataset, size of your choice
    Calls the 'generate()' method from run_gans.py
        Each generate() call creates 128 datapoints
    """
    args = get_args()
    masterImage = generate(BATCH_SIZE=args.batch_size, nice=args.nice)

    for i in range(10): #1408 data points
         images_i = generate(BATCH_SIZE=args.batch_size, nice=args.nice)
         masterImage = np.concatenate([masterImage, images_i], axis=0)

    print("LOOK HERE", type(masterImage))
    print(masterImage.shape)
    np.save("fake_mnist_dataset", masterImage, allow_pickle = True)

    image = combine_images(masterImage)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
         "final_generated_image.png")

if __name__ == "__main__":
    generate_data_set()
