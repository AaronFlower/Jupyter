# -*- coding: utf-8 -*-
import numpy as np
import random

class Network(object):
    '''
    A Neural network using mini-batch-size stochastic gradient descent.
    '''
    def __init__(self, sizes):
        '''
        Construct a NN with layers and init its weights, and bias
        '''
        self.L = len(sizes)
        self.weights = [np.random.randn(sizes[i + 1], sizes[i]) for i in range(self.L - 1)]
        self.bias = [np.random.randn(sizes[i + 1], 1) for i in range(self.L - 1)]
        return

    def train(self, train_data, epoches, alpha, mini_batch_size, test_data = None):
        '''
        Using stochastic gradient descent to train NN model
        '''
        print("Train begin:")
        M = len(train_data)
        if test_data : test_N = len(test_data)
        k = mini_batch_size
        for i in range(epoches):
            random.shuffle(train_data)
            for j in range(0, M, k):
                mini_batch_data = train_data[j:j+k]
                self.mini_batch_update(mini_batch_data, alpha)
            if (test_data):
                print("Epoch {0} : {1} / {2} ".format(i + 1, self.evaluate(test_data), test_N))

        return

    def mini_batch_update(self, batch_data, alpha):
        '''
        Update weights for every example in batch_data.
        '''
        hidden_layers = self.L - 1
        nabla_ws = [np.zeros(self.weights[i].shape) for i in range(hidden_layers)]
        nabla_bs = [np.zeros(self.bias[i].shape) for i in range(hidden_layers)]

        M = len(batch_data)
        for i in range(M):
            X, y = batch_data[i]
            nabla_w, nabla_b = self.BP(X, y)
            nabla_ws = [nabla_ws[i] + nabla_w[i] for i in range(hidden_layers)]
            nabla_bs = [nabla_bs[i] + nabla_b[i] for i in range(hidden_layers)]

        # update weights and biases
        m_i = 1.0 / M
        self.weights = [self.weights[i] - alpha * m_i * nabla_ws[i] for i in range(hidden_layers)]
        self.bias = [self.bias[i] - alpha * m_i * nabla_bs[i] for i in range(hidden_layers)]
        return

    def BP(self, X, y):
        '''
        Backpropogation method
        '''
        hidden_layers = self.L - 1
        nabla_w = [np.zeros(self.weights[i].shape) for i in range(hidden_layers)]
        nabla_b = [np.zeros(self.bias[i].shape) for i in range(hidden_layers)]

        a = X
        cache_z = []
        cache_a = [a]

        for i in range(hidden_layers):
            z = np.dot(self.weights[i], a) + self.bias[i]
            a = self.sigmoid(z)
            cache_z.append(z)
            cache_a.append(a)

        delta = (a - y) * self.sigmoid_prime(z)
        nabla_w[-1] = np.dot(delta, cache_a[-2].T)
        nabla_b[-1] = delta

        for i in range(2, self.L):
            z = cache_z[-i]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[- i + 1].T, delta) * sp
            nabla_w[- i] = np.dot(delta, cache_a[- i - 1].T)
            nabla_b[- i] = delta
        return nabla_w, nabla_b


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, y):
        t = self.sigmoid(y)
        return t * (1 - t)

    def FP(self, a):
        for i in range(self.L - 1):
            z = np.dot(self.weights[i], a) + self.bias[i]
            a = self.sigmoid(z)
        return np.argmax(a)

    def evaluate(self, test_data):
        return np.sum([int(self.FP(X) == y) for X, y in test_data])





