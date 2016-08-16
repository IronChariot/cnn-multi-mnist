# Code to combine MNIST 28x28 images into images containing 2 to 4 MNIST images within a single IMAGE_LENGTHx28 image.
# Need to create label for each image. If possible, keep same format as original MNIST data.

# Initial MNIST to numpy code adapted from Gustav Sweyla (http://g.sweyla.com/blog/2012/mnist-numpy/)

import os
import struct
import numpy as np
import random
from array import array as pyarray
from numpy import array, int8, uint8, zeros

IMAGE_LENGTH = 168


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

    # For our 60,000 samples,
    for sample in range(60000):
        # 1) Select a number of labels to include (2 to 4):
        n_labels = 2 + random.randrange(3)
        # 2) Randomly select that number of images from MNIST training set:
        rand_indices = []
        for i in range(n_labels):
            rand_indices.append(random.randrange(60000))
        # 3) Easy part, create the new label:
        new_label = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for index in rand_indices:
            new_label[labels[index][0]] = 1.0
        new_label = np.array(new_label)
        # 4) Harder part, create new image
        #   a) Choose the starting positions for however many images we're copying in
        #      Select x random positions, check if they're allowed, if not, try again
        invalid = True
        while invalid:
            start_pos = []
            invalid = False
            for i in range(n_labels):
                start_pos.append(random.randrange(IMAGE_LENGTH - 28))
            for i in range(1, n_labels):
                if abs(start_pos[i] - start_pos[i - 1]) < 28:
                    invalid = True
        # 5) Hardest part, actual copy in the images where they are needed:
        new_image = np.zeros([28, IMAGE_LENGTH])



