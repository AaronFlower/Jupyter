# -*- coding: utf-8 -*-

import random
import numpy as np

class Cost(object):
    def __init__(self):
        pass

    def cost(a, y):
        raise NotImplementedError('Subclass must implement this cost abstract method')

    def delta(a, y, z, activate_prime):
        raise NotImplementedError('Subclass must implement this delta abstract method for output layer')

class QuadraticCost(Cost):
    def cost(a, y):
        '''
            Quadratic Cost with an output `a` and desired output `y`
        '''
        return 0.5 * (a - y)

    def delta(a, y, z, activate_prime):
        '''
            Compute the output delta
        '''
        return (a - y) * activate_prime(z)

class CrossEntropyCost(Cost):
    def cost(a, y):
        base = -[y * np.log(a) + (1 - y) * np.log(1 - a)]
        return np.sum(np.nan_to_num(base))

    def delta(a, y, z, activate_prime):
        return (a - y)

class NeuralNetwork (object):
    '''
    A simple neural network
    '''
    def __init__(self, sizes, cost = CrossEntropyCost):
        '''
        Initialize the weights and biases
        '''
        self.num_layers = len(sizes)
        self.hidden_layers = self.num_layers - 1
        self.sizes = sizes
        self.cost = cost
        self.default_weight_initailizer()

    def default_weight_initailizer(self):
        '''
        Use Guassian distribution to initialize weights and biases with respecting
        the previous layers units.
        '''
        self.weights = [np.random.randn(self.sizes[i], self.sizes[i - 1]) / np.sqrt(self.sizes[i - 1])
                        for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(self.sizes[i], 1) for i in range(1, self.num_layers)]

    def large_weight_initializer(self):
        '''
        Use randn to initialize weiths and biases
        '''
        self.weights = [np.random.randn(self.sizes[i], self.sizes[i - 1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(self.sizes[i], 1) for i in range(1, self.num_layers)]

    def activate(self, z):
        '''
        Activate function
        '''
        return 1.0 / (1 + np.exp(-z))

    def activate_prime(self, z):
        '''
        Actiate prime function
        '''
        f = 1.0 / (1 + np.exp(-z))
        return f * (1 - f)

    def fp(self, x):
        '''
        Forward propagation to predict the output
        '''
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activate(z)
        return a

    def evaluate(self, test_data):
        '''
        To compute the model accuracy
        '''
        results = [(np.argmax(self.fp(x)), y) for x, y in test_data]
        return np.sum([int(yhat == y) for yhat, y in results])

    def model(self, train_data, epoches, alpha, mini_batch_size, lmbda, test_data = None):
        '''
        To train the NN model
        '''
        if test_data: ntest = len(test_data)
        n = len(train_data)

        for i in range(epoches):
            random.shuffle(train_data)
            mini_batches = [train_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alpha, lmbda, n)

            if test_data:
                print('Epoch {0}: {1}/{2}'.format(i, self.evaluate(test_data), ntest))
            else:
                print('Epoch {0} complete!'.format(i))

        return

    def update_mini_batch(self, mini_batch, alpha, lmbda, n):
        '''
        Using mini batch gradient descent to update weights and biases.
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        nbatch = len(mini_batch)

        for X, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.bp(X, y)

            nabla_w = [w + dw for w, dw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [b + db for b, db in zip(nabla_b, delta_nabla_b)]

        anb = alpha / nbatch
        aln = alpha * lmbda / n
        self.weights = [w - aln * w - anb * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - anb * nb for b, nb in zip(self.biases, nabla_b)]

        return

    def bp(self, X, y):
        '''
        Use backpropagation to compute weights and biases for every layers
        '''
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        a = X
        cache_a = [a]
        cache_z = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = self.activate(z)
            cache_z.append(z)
            cache_a.append(a)

        delta = self.cost.delta(a, y, z, self.activate_prime)
        nabla_w[-1] = np.dot(delta, cache_a[-2].T)
        nabla_b[-1] = delta

        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activate_prime(cache_z[-l])
            nabla_w[-l] = np.dot(delta, cache_a[-l -1].T)
            nabla_b[-l] = delta
        return nabla_w, nabla_b

