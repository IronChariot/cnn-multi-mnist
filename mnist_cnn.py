"""
Convolutional Neural Network code, experiment to see how well we can handle inputs with multiple labels, 
and if we can get the order in which the labelled images appear in the long input image.

Code adapted from Aymeric Damien's CNN Tensorflow example (https://github.com/aymericdamien/TensorFlow-Examples/)
Uses data import code from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/

First, we'll train the network to recognise MNIST digits in the middle of long (wide) images
Second, we'll continue to train the same network with MNIST digits in random locations in long images
Finally, we'll continue to train the same network with multiple random MNIST digits in random locations

"""

import tensorflow as tf
import os


HEIGHT = 28
LENGTH = 28
BATCH_SIZE = 128
DATA_PATH = "./base_mnist"
TRAIN_FILENAME = "train_mnist_small_single.tfrecords"
TEST_FILENAME = "test_mnist_small_single.tfrecords"


def read_and_decode_single_example(filename):
    # first construct a queue containing a list of filenames.
    # this lets a user split up their dataset in multiple files to keep
    # size down
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    # Unlike the TFRecordWriter, the TFRecordReader is symbolic
    reader = tf.TFRecordReader()
    # One can read a single serialized example from a filename
    # serialized_example is a Tensor of type string.
    _, serialized_example = reader.read(filename_queue)
    # The serialized example is converted back to actual values.
    # One needs to describe the format of the objects to be returned
    features = tf.parse_single_example(
        serialized_example,
        features={
            # We know the length of both fields. If not the
            # tf.VarLenFeature could be used
            'label': tf.FixedLenFeature([10], tf.int64),
            'image': tf.FixedLenFeature([HEIGHT * LENGTH], tf.int64)
        })
    # now return the converted data
    label = tf.cast(features['label'], tf.float32)
    image = tf.cast(features['image'], tf.float32) / 255.0
    return label, image


# get single examples
label, image = read_and_decode_single_example(os.path.join(DATA_PATH, TRAIN_FILENAME))
# groups examples into batches randomly
images_batch, labels_batch = tf.train.shuffle_batch(
    [image, label], batch_size=BATCH_SIZE,
    capacity=2000,
    min_after_dequeue=1000)

# get single test example
test_label, test_image = read_and_decode_single_example(os.path.join(DATA_PATH, TEST_FILENAME))
# groups examples into batches randomly
test_images_batch, test_labels_batch = tf.train.shuffle_batch(
    [test_image, test_label], batch_size=BATCH_SIZE,
    capacity=2000,
    min_after_dequeue=1000)

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = BATCH_SIZE
display_step = 10

# Network Parameters
n_input = HEIGHT * LENGTH  # MNIST data input (img shape: 28*168)
n_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
x = images_batch
y = labels_batch
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, HEIGHT, LENGTH, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    pool1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    pool2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    reshaped_pool2 = tf.reshape(pool2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(reshaped_pool2, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([(HEIGHT // 4) * (LENGTH // 4) * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={keep_prob: 1.})
            print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={keep_prob: 1.})
