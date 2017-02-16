'''
A Multilayer Perceptron implementation example

This example is using the MNIST database of handwritten digits from Tensorflow
(http://yann.lecun.com/exdb/mnist/)

Author: Tony Hong
'''

# Import MNIST data
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
rng = np.random.RandomState(1)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 1
display_step = 1

# Network Parameters
n_hidden = 256 # 1st layer number of features
n_in = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# init layers weight & bias
W = {
    'h': np.asarray(
            rng.uniform(
                # low=-np.sqrt(6. / (n_in + n_hidden)),
                # high=np.sqrt(6. / (n_in + n_hidden)),
                low=-0.1,
                high=0.1,
                size=(n_in, n_hidden)
            ),
            dtype='f'
        ),
    'o': np.asarray(
            rng.uniform(
                low=-0.1,
                high=0.1,
                size=(n_hidden, n_classes)
            ),
            dtype='f'
        )
}

b = {
    'h': np.zeros(
            (1, n_hidden),
            dtype='f'
        ),
    'o': np.zeros(
            (1, n_classes),
            dtype='f'
        )
}


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

### Maths
def sigmoid(x):
    y = 1. / (1 + np.exp(-x))
    return y

def softmax(x):
    y = np.exp(x) / np.exp(x).sum()
    return y

def MSE(y_o, y):
    return 0.5 * (y_o - y).dot((y_o - y).T)

def dMSE_dWo(y_o, y, h):
    return h.T.dot(y_o - y)

def dMSE_dbo(y_o, y):
    return y_o - y

def dMSE_dWh(y_o, y, h, W_o, x):
    return (1. - h).dot(h.T) * x.T.dot(y_o - y).dot(W_o.T)

def dMSE_dbh(y_o, y, h, W_o):
    return (1. - h).dot(h.T) * (y_o - y).dot(W_o.T)

### Create model
def hidden_layer_out(x, W, b):
    # Hidden layer with RELU activation
    linlay = np.dot(x, W) + b
    h = sigmoid(linlay)
    return h

def output_layer_out(h, W_o, b_o):
    # Output layer with linear activation
    out_layer = np.dot(h, W_o) + b_o
    normalized_output = softmax(out_layer)
    output = np.zeros(normalized_output.shape[1])
    output[np.argmax(normalized_output)] = 1
    return output

def hidden_layer_bp(y_o, y, h, W_o, x, epsilon):
    W['h'] = W['h'] - epsilon * dMSE_dWh(y_o, y, h, W_o, x)
    b['h'] = b['h'] - epsilon * dMSE_dbh(y_o, y, h, W_o)

def output_layer_bp(y_o, y, h, epsilon):
    W['o'] = W['o'] - epsilon * dMSE_dWo(y_o, y, h)
    b['o'] = b['o'] - epsilon * dMSE_dbo(y_o, y)


# 5.2.a)
def predict(x):
    hidden_layer = hidden_layer_out(x, W['h'], b['h'])
    y_o = output_layer_out(hidden_layer, W['o'], b['o'])
    return hidden_layer, y_o

def train():
    train_error_list = list()
    error_list = list()

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        correct = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            x, y = mnist.train.next_batch(batch_size)

            # Construct model
            h, y_o = predict(x)

            # Define loss
            cost = MSE(y_o, y)

            # Run optimization op (backprop) and cost op (to get loss value)
            output_layer_bp(y_o, y, h, learning_rate)
            hidden_layer_bp(y_o, y, h, W['o'], x, learning_rate)

            # Compute average loss
            avg_cost += float(cost) / total_batch

            if np.argmax(y_o) == np.argmax(y):
                correct += 1
            # print("Batch:", '%04d' % (i+1), "cost=", \
            #     "{:.9s}".format(cost))

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

        # Calculate accuracy
        train_accuracy = correct / total_batch
        print("Train Accuracy:", train_accuracy)
        train_error = 1. - train_accuracy
        train_error_list.append(train_error)

        accuracy = eval(mnist.test.images, mnist.test.labels)
        print("Test Accuracy:", accuracy)
        error = 1. - accuracy
        error_list.append(error)

    print("Optimization Finished!")
    return train_error_list, error_list

# Test model
def eval(xlist, ylist):
    correct = 0
    N = len(xlist)
    for i in range(N):
        h, y_o = predict(xlist[i].reshape(1, n_in))
        if np.argmax(y_o) == np.argmax(ylist[i]):
            correct += 1
    accuracy = float(correct) / N
    print correct, N, accuracy
    return accuracy

print("Init. eval.:", eval(mnist.test.images, mnist.test.labels))

train_error_list, error_list = train()

plot2D([x + 1 for x in range(training_epochs)], train_error_list, 'epoch_train_error', Xlabel='Epoch', Ylabel='Train Error', title='Epoch vs. Train Error')
plot2D([x + 1 for x in range(training_epochs)], error_list, 'epoch_test_error', Xlabel='Epoch', Ylabel='Test Error', title='Epoch vs. Test Error')
