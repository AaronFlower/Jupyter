# -*- coding: utf-8 -*-

import numpy as np
import random

class NeuralNetwork (object):
    '''
    A simple nueral network
    '''

    def __init__ (self, sizes):
        '''
        Initialize the weights and biases
        '''
        self.layers = len(sizes)
        self.weights = [np.random.randn(sizes[i], sizes[i - 1]) for i in range(1, self.layers)]
        self.biases = [np.random.randn(sizes[i], 1) for i in range(1, self.layers)]

    def FP(self, a):
        '''
        Use a as input parameter to get output as a prediction.
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activate(z)

        return a

    def model(self, train_data, epoches, alpha, mini_batch_sizes, test_data = None):
        '''
        Use mini stochastic gradient descent to train the NeuralNetwork model
        '''
        if test_data: n_test = len(test_data)
        n = len(train_data)

        for i in range(epoches):
            data = random.shuffle(train_data)
            batches = [train_data[k : k + mini_batch_sizes] for k in range(0, n, mini_batch_sizes)]

            for mini_batch in batches:
                self.update_mini_batch(mini_batch, alpha)
            if test_data:
                print('Epoch {0}: {1}/{2}'.format(i, self.evaluate(test_data), n_test))
            else:
                print('Epoch {0} complete'.format(i))

        return

    def update_mini_batch(self, mini_batch, alpha):
        '''
        Use BP to update weights and biases
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # accumulate gradients
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.BP(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        am = alpha * ( 1 / len(mini_batch))
        self.weights = [w - am * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - am * nb for b, nb in zip(self.biases, nabla_b)]
        return

    def BP(self, x, y):
        '''
        Get Nabla_b, Nabla_w using BP.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # FP
        a = x
        cache_a = [x] # list to store all the activations for all layers
        cache_z = [] # list to store all the z vectors for all layers

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = self.activate(z)
            cache_z.append(z)
            cache_a.append(a)

        # Compute output layer parameters.
        delta = self.cost_derivative(cache_a[-1], y) * self.activate(cache_z[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, cache_a[-2].T)

        # Compute hidden layer parameters
        for l in range(2, self.layers):
            z = cache_z[-l]
            actp = self.activate_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * actp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, cache_a[-l - 1].T)
        return (nabla_b, nabla_w)


    def cost_derivative(self, output_a, y):
        '''
        Get output error
        '''
        return output_a - y

    def evaluate(self, test_data):
        '''
        count predictions
        '''
        test_results = [(np.argmax(self.FP(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for x,y in test_results)

    def activate(self, z):
        '''
        activation function
        '''
        return 1 / (1 + np.exp(-z))

    def activate_prime(self, z):
        '''
        activation prime function
        '''
        f = 1 / (1 + np.exp(-z))
        return f * (1 - f)

