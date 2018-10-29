# -*- coding: utf-8 -*-

import random
import numpy as np

class NeuralNetwork (object):
    '''
    A simple neural network
    '''

    def __init__(self, sizes):
        '''
        initialize the weights and biases
        '''
        self.layers = len(sizes)
        self.hidden_layers = self.layers - 1
        self.weights = [np.random.randn(sizes[i], sizes[i - 1]) for i in range(1, self.layers)]
        self.biases = [np.random.randn(sizes[i], 1) for i in range(1, self.layers)]

        return

    def activate(self, z):
        '''
        activate function
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
        Evaluate the model accuracy
        '''
        test_results = [(np.argmax(self.fp(x)), y) for x, y in test_data]
        return sum([int(yhat == y) for yhat, y in test_results])

    def model(self, train_data, epoches, alpha, mini_batch_size, test_data = None):
        '''
        Train the model
        '''
        if test_data: ntest = len(test_data)
        n = len(train_data)
        for i in range(epoches):
            random.shuffle(train_data)
            mini_batches = [train_data[k: k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha)

            if test_data:
                print('Epoch {0}: {1}/{2}'.format(i, self.evaluate(test_data), ntest))
            else:
                print('Epoch {0} complete!'.format(i))
        return

    def update_mini_batch(self, mini_batch, alpha):
        '''
        Using stochastic mini batch to update weighs and biases
        '''
        n = len(mini_batch)
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.bp(x, y)

            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        an = alpha / n
        for l in range(self.hidden_layers):
            self.weights[l] = self.weights[l] - an * nabla_w[l]
            self.biases[l] = self.biases[l] - an * nabla_b[l]
        return

    def bp(self, x, y):
        '''
        Backpropagation method
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # fp
        a = x
        cache_a = [a]
        cache_z = []
        for l in range(self.hidden_layers):
            z = np.dot(self.weights[l], a) + self.biases[l]
            a = self.activate(z)
            cache_a.append(a)
            cache_z.append(z)

        # bp
        # Compute the output layer delta
        delta = -(y - a) * self.activate_prime(z)
        nabla_w[-1] = np.dot(delta, cache_a[-2].T)
        nabla_b[-1] = delta

        # Compute the hidden layer delta
        for l in range(2, self.layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activate_prime(cache_z[-l])
            nabla_w[-l] = np.dot(delta, cache_a[-l - 1].T)
            nabla_b[-l] = delta
        return nabla_w, nabla_b

