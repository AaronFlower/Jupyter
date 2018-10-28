# -*- coding: utf-8 -*-

import numpy as np
import random

class NeuralNetwork(object):
    '''
    A simple Neural Network
    '''

    def __init__(self, sizes):
        '''
        Initialize layers, weights, biases
        '''
        self.layers = len(sizes)
        self.weights = [np.random.randn(sizes[i], sizes[i - 1]) for i in range(1, self.layers)]
        self.biases = [np.zeros((sizes[i], 1)) for i in range(1, self.layers)]
        return

    def activate(self, z):
        '''
        use simple sigmoid function to implement activate function
        '''
        return 1 / (1 + np.exp(-z))

    def activate_prime(self, z):
        '''
        activate prime function
        '''
        f = 1 / (1 + np.exp(-z))
        return f * (1 - f)

    def fp(self, x):
        '''
        use fp to predict the output
        '''
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activate(z)
        return a

    def evaluate(self, test_data):
        '''
        To evaluate the out accuracy
        '''
        test_results = [(np.argmax(self.fp(x)), y) for x, y in test_data]
        return sum([int(yhat == y) for y, yhat in test_results])

    def model(self, train_data, epoches, alpha, batch_size, test_data = None):
        '''
        Use stochastic batch update methods to train a nn model
        '''
        if test_data: test_n = len(test_data)
        n = len(train_data)

        for i in range(epoches):
            random.shuffle(train_data)
            mini_batches = [train_data[k : k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)

            if test_data:
                print('Epoche {0}: {1} / {2}'.format(i, self.evaluate(test_data), test_n))
            else:
                print('Epoche {0} complete'.format(i))
        return

    def update_mini_batch(self, mini_batch, alpha):
        '''
        Using mini batch example to update weights, and bias
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        n = len(mini_batch)

        for example in mini_batch:
            delta_nabla_w, delta_nabla_b = self.bp(example)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        an = alpha / n
        for l in range(self.layers - 1):
            self.weights[l] = self.weights[l] - an * nabla_w[l]
            self.biases[l] = self.biases[l] - an * nabla_b[l]

        return

    def bp(self, example):
        '''
        To implete bp function
        '''
        x, y = example
        hidden_layers = self.layers - 1
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        a = x
        cache_a = [a]
        cache_z = []

        # fp
        for i in range(hidden_layers):
            z = np.dot(self.weights[i], a) + self.biases[i]
            a = self.activate(z)
            cache_z.append(z)
            cache_a.append(a)

        # bp
        # compute output layers nablas.
        delta = -(y - a) * self.activate_prime(z)
        nabla_w[-1] = np.dot(delta, cache_a[-2].T)
        nabla_b[-1] = delta

        # compute the hidden layers nablas.
        for l in range(2, self.layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activate_prime(cache_z[-l])
            nabla_w[-l] = np.dot(delta, cache_a[-l - 1].T)
            nabla_b[-l] = delta

        return nabla_w, nabla_b
