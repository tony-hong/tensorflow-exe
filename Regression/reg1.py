'''
A linear regression learning algorithm example using TensorFlow library.
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.2
training_epochs = 20000
display_step = 50

# Training Data
# train_X = 1./numpy.asarray([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8])
# train_Y = 1./numpy.asarray([0.0042, 0.0078, 0.0146, 0.0189, 0.0194, 0.0259, 0.0279])
train_X = numpy.asarray([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8])
train_Y = numpy.asarray([0.0042, 0.0078, 0.0146, 0.0189, 0.0194, 0.0259, 0.0279])
n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float64")
Y = tf.placeholder("float64")

# Set model weights
W = tf.Variable(numpy.array([0.05277032]), name="weight", dtype='float64')
b = tf.Variable(numpy.array([0.00483934]), name="bias", dtype='float64')

# Construct a linear model
pred = tf.divide(tf.multiply(X, W), tf.add(X, b))

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
# Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    # plt.plot(train_X, (sess.run(W) * train_X + sess.run(b)), label='Fitted line')
    plt.plot(train_X, (sess.run(W) * train_X) / (train_X + sess.run(b)), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    # test_X = 1./numpy.asarray([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8])
    # test_Y = 1./numpy.asarray([0.0042, 0.0078, 0.0146, 0.0189, 0.0194, 0.0259, 0.0279])
    test_X = numpy.asarray([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8])
    test_Y = numpy.asarray([0.0042, 0.0078, 0.0146, 0.0189, 0.0194, 0.0259, 0.0279])

    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(test_X, (sess.run(W) * train_X + sess.run(b)), label='Fitted line')
    plt.plot(test_X, (sess.run(W) * train_X) / (train_X + sess.run(b)), label='Fitted line')
    plt.legend()
    plt.show()

    # print("Result: ", testing_cost, "V_max=", 1.0/sess.run(b), "K_m=", 1.0 / sess.run(b) / sess.run(W), '\n')
    print("Result: ", testing_cost, "V_max=", sess.run(W), "K_m=", sess.run(b), '\n')
