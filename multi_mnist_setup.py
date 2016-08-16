# Code to combine MNIST SIZExSIZE images into images containing MIN to MAX MNIST images within a single IMAGE_LENGTHxSIZE image.
# Need to create label for each image. If possible, keep same format as original MNIST data.

# Initial MNIST to numpy code adapted from Gustav Sweyla (http://g.sweyla.com/blog/2012/mnist-numpy/)

import os
import struct
import numpy as np
import random
from array import array as pyarray
from numpy import array, int8, uint8, zeros

MIN = 2
MAX = 4
SIZE = 28
IMAGE_LENGTH = (MAX + 2) * SIZE
MNIST_SAMPLES = 60000


def load_mnist(dataset="training", digits=np.arange(10), path="./base_mnist"):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    n = len(ind)

    images = zeros((n, rows, cols), dtype=uint8)
    labels = zeros((n, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

if __name__ == "__main__":
    random.seed(1234)

    # Extract images and labels from base MNIST files
    images, labels = load_mnist()
    new_images = []
    new_labels = []

    # For our 60,000 samples, (100 to start with for testing!)
    for sample in range(100):
        # 1) Select a number of labels to include (MIN to MAX):
        n_labels = MIN + random.randrange(1 + (MAX - MIN))
        # 2) Randomly select that number of samples from MNIST training set:
        rand_indices = [random.randrange(MNIST_SAMPLES) for i in range(n_labels)]
        # 3) Easy part, create the new label:
        new_label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for index in rand_indices:
            new_label[labels[index][0]] = 1.0
        new_label = np.array(new_label)
        # 4) Choose the starting positions for however many images we're copying in
        #    Select n_labels random positions, check if they're allowed, if not, try again
        #    Since we only choose starting positions not too close to the end, just need to check their
        #    proximity to the other starting positions.
        invalid = True
        while invalid:
            start_positions = [random.randrange(IMAGE_LENGTH - SIZE) for i in range(n_labels)]
            invalid = False
            for i in range(1, n_labels):
                if abs(start_positions[i] - start_positions[i - 1]) < SIZE:
                    invalid = True
        # 5) Last part, actually copy in the images where they are needed:
        new_image = np.zeros([SIZE, IMAGE_LENGTH])

        for start_pos, index in zip(start_positions, rand_indices):
            for row in range(SIZE):
                for col in range(SIZE):
                    new_image[row][start_pos + col] = images[index][row][col]
            
        new_images.append(new_image)
        new_labels.append(new_label)
        
    new_images = np.array(new_images)
    new_labels = np.array(new_labels)
    
    np.save("multi_inputs.npy", new_images)
    np.save("multi_outputs.npy", new_labels)
