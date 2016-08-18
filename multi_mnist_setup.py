# Code to combine MNIST SIZExSIZE images into images containing MIN to MAX MNIST images within
# a single IMAGE_LENGTHxSIZE image. Need to create label for each image.
# If possible, keep same format as original MNIST data.

# Initial MNIST to numpy code adapted from Gustav Sweyla (http://g.sweyla.com/blog/2012/mnist-numpy/)

import os
import struct
import numpy as np
import tensorflow as tf
import random
from array import array as pyarray
from numpy import array, int8, uint8, zeros

MIN = 2
MAX = 4
SIZE = 28
IMAGE_LENGTH = (MAX + 2) * SIZE
MNIST_SAMPLES = 60000
DATA_PATH = "./base_mnist"


def load_mnist(dataset="training", digits=np.arange(10)):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(DATA_PATH, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(DATA_PATH, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(DATA_PATH, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(DATA_PATH, 't10k-labels.idx1-ubyte')
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


def create_small_single_mnist(samples=60000, dataset="training"):
    """ Create a dataset where a single MNIST digit is located in the middle of a long image. """
    # Extract images and labels from base MNIST files
    images, labels = load_mnist(dataset)
    new_images = []
    new_labels = []

    if samples > len(images):
        print "There aren't that many images in the MNIST {} dataset!".format(dataset)
        return

    # For however many samples...
    for sample in range(samples):
        # Create the new label (one hot encoding):
        new_label = np.zeros([10])
        new_label[labels[sample][0]] = 1.0
        # Create the new image:
        new_image = np.zeros([SIZE, SIZE])
        start_position = 0

        for row in range(SIZE):
            for col in range(SIZE):
                new_image[row][start_position + col] = images[sample][row][col]

        new_image = new_image.flatten()

        new_images.append(new_image)
        new_labels.append(new_label)

    new_images = np.array(new_images)
    new_labels = np.array(new_labels)

    return new_images, new_labels


def create_single_mnist(samples=60000, dataset="training", noise=False):
    """ Create a dataset where a single MNIST digit is located in the middle of a long image. """
    # Extract images and labels from base MNIST files
    images, labels = load_mnist(dataset)
    new_images = []
    new_labels = []

    if samples > len(images):
        print "There aren't that many images in the MNIST {} dataset!".format(dataset)
        return

    # For however many samples...
    for sample in range(samples):
        # Create the new label (one hot encoding):
        new_label = np.zeros([10])
        new_label[labels[sample][0]] = 1.0
        # Create the new image:
        if noise:
            new_image = np.random.randint(256, size=(SIZE, IMAGE_LENGTH))
        else:
            new_image = np.zeros([SIZE, IMAGE_LENGTH])
        start_position = (IMAGE_LENGTH // 2) - (SIZE // 2)

        for row in range(SIZE):
            for col in range(SIZE):
                new_image[row][start_position + col] = images[sample][row][col]

        new_image = new_image.flatten()

        new_images.append(new_image)
        new_labels.append(new_label)

    new_images = np.array(new_images)
    new_labels = np.array(new_labels)

    return new_images, new_labels


def create_rand_single_mnist(samples=60000, dataset="training", noise=False):
    """ Create a dataset where a single MNIST digit is located in a random location in a long image. """
    # Extract images and labels from base MNIST files
    images, labels = load_mnist(dataset)
    new_images = []
    new_labels = []

    if samples > len(images):
        print "There aren't that many images in the MNIST {} dataset!".format(dataset)
        return

    # For however many samples...
    for sample in range(samples):
        # Create the new label (one hot encoding):
        new_label = np.zeros([10])
        new_label[labels[sample][0]] = 1
        # Choose the starting positions number image.
        start_position = random.randrange(IMAGE_LENGTH - SIZE)
        # Copy in the image where it is needed:
        if noise:
            new_image = np.random.randint(256, size=(SIZE, IMAGE_LENGTH))
        else:
            new_image = np.zeros([SIZE, IMAGE_LENGTH])

        for row in range(SIZE):
            for col in range(SIZE):
                new_image[row][start_position + col] = images[sample][row][col]

        new_image = new_image.flatten()

        new_images.append(new_image)
        new_labels.append(new_label)

    new_images = np.array(new_images)
    new_labels = np.array(new_labels)

    return new_images, new_labels


def create_rand_multi_mnist(samples=60000, dataset="training", noise=False):
    """ Create a dataset where multiple (MIN to MAX) random MNIST digits are randomly located in a long image. """
    # Extract images and labels from base MNIST files
    images, labels = load_mnist(dataset)
    new_images = []
    new_labels = []

    # For however many samples...
    for sample in range(samples):
        # 1) Select a number of labels to include (MIN to MAX):
        n_labels = MIN + random.randrange(1 + (MAX - MIN))
        # 2) Randomly select that number of samples from MNIST training set:
        rand_indices = [random.randrange(MNIST_SAMPLES) for i in range(n_labels)]
        # 3) Create the new label:
        new_label = np.zeros([10])
        for index in rand_indices:
            new_label[labels[index][0]] = 1.0
        # 4) Choose the starting positions for however many images we're copying in
        #    Select n_labels random positions, check if they're allowed, if not, try again
        #    Since we only choose starting positions not too close to the end, just need to check their
        #    proximity to the other starting positions.
        invalid = True
        start_positions = []
        while invalid:
            start_positions = [random.randrange(IMAGE_LENGTH - SIZE) for i in range(n_labels)]
            invalid = False
            for i in range(1, n_labels):
                if abs(start_positions[i] - start_positions[i - 1]) < SIZE:
                    invalid = True
        # 5) Last part, actually copy in the images where they are needed:
        if noise:
            new_image = np.random.randint(256, size=(SIZE, IMAGE_LENGTH))
        else:
            new_image = np.zeros([SIZE, IMAGE_LENGTH])

        for start_pos, index in zip(start_positions, rand_indices):
            for row in range(SIZE):
                for col in range(SIZE):
                    new_image[row][start_pos + col] = images[index][row][col]

        new_image = new_image.flatten()

        new_images.append(new_image)
        new_labels.append(new_label)
        
    new_images = np.array(new_images)
    new_labels = np.array(new_labels)

    return new_images, new_labels


if __name__ == "__main__":
    random.seed(1234)

    images, labels = create_small_single_mnist(dataset="training", samples=60000)

    writer = tf.python_io.TFRecordWriter(os.path.join(DATA_PATH, "train_mnist_small_single.tfrecords"))
    for example_id in range(images.shape[0]):
        features = images[example_id]
        label = labels[example_id]

        # Construct example proto object
        example = tf.train.Example(
            # Example contains a Features proto object
            features=tf.train.Features(
                # Features contains a map of string to Feature proto objects
                feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=label.astype("int64"))),
                    'image': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=features.astype("int64"))),
                }
            )
        )
        # use the proto object to serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized object to disk
        writer.write(serialized)

