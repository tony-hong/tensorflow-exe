'''
A convolutional neural network (CNN) implementation example using TensorFlow library.

This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Tony Hong
'''

# Import MNIST data
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 1e-4
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_size = 28
n_mask = 5
n_hidden = 32 # 1st layer number of features
n_fully = 1024
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    '1': tf.Variable(tf.random_normal([n_mask, n_mask, 1, n_hidden])),
    '2': tf.Variable(tf.random_normal([n_size * n_size * n_hidden / 4, n_fully])),
    'out': tf.Variable(tf.random_normal([n_fully, n_classes]))
}
biases = {
    '1': tf.Variable(tf.constant(0.1, shape=[n_hidden])),
    '2': tf.Variable(tf.constant(0.1, shape=[n_fully])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Create model
def get_layer_1(x, weights, biases):
    x_matrix = tf.reshape(x, [-1, n_size, n_size, 1])
    h_conv_1 = conv2d(x_matrix, weights['1']) + biases['1']
    # Hidden layer with RELU activation
    h_nll_1 = tf.nn.relu(h_conv_1)
    h_pool_1 = max_pool_2x2(h_nll_1)
    return h_pool_1

def get_layer_2(x, weights, biases):
    h_pool_flat = tf.reshape(x, [-1, n_size * n_size * n_hidden / 4])
    h_fc = tf.add(tf.matmul(h_pool_flat, weights['2']), biases['2'])

    # Hidden layer with RELU activation
    h_nll_2 = tf.nn.relu(h_fc)
    return h_nll_2

def get_output_layer(x, weights, biases):
    # Output layer with linear activation
    out_layer = tf.matmul(x, weights['out']) + biases['out']
    return out_layer

def predict(x, weights, biases):
    layer_1 = get_layer_1(x, weights, biases)
    layer_2 = get_layer_2(layer_1, weights, biases)
    y_p = get_output_layer(layer_2, weights, biases)
    
    # y_p = get_output_layer(layer_1, weights, biases)
    return y_p



### Plots
def plot2D(X, Y, fname, Xlabel='X', Ylabel='Y', title='plot'):
    '''plot for 2D
    '''
    N = len(X)

    xarray = np.array(X)
    yarray = np.array(Y)    

    plt.figure()
    plt.plot(xarray, yarray, '-')
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    plt.title(title)
    if fname:
        plt.savefig(fname + '.png')
    # plt.show()
    plt.close()


if __name__=="__main__":
    # Construct model
    pred = predict(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Training cost list
    cost_list = list()

    # Launch the graph
    with tf.Session() as sess:
        
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
            cost_list.append(avg_cost)

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        plot2D([x + 1 for x in range(training_epochs)], cost_list, 'epoch_train_cost', Xlabel='Epoch', Ylabel='Train Cost', title='Epoch vs. Train Cost')
