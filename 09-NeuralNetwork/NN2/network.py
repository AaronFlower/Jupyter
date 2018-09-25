# -*- coding: utf-8 -*-

import numpy as np
import random



class NueralNetwork(object):
    '''
    A simple Nueral Network
    '''

    def __init__(self, sizes):
        self.L = len(sizes)
        self.biases = [np.random.randn(nL, 1) for nL in sizes[1:]]
        self.weights = [np.random.randn(nL, nPreL)
                        for nL, nPreL in zip(sizes[1:], sizes)]

    def sigmoid(self, z):
        '''
            激活函数
        '''
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        '''
            激活函数的一阶导数
        '''
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def train(self, train_data, epochs, alpha, mini_batch_size, test_data = None):
        '''
            根据 Mini-batch stochastic gradient descent 来训练模型。
        '''
        m = len(train_data)
        if test_data: test_m = len(test_data)
        k = mini_batch_size

        for i in range(epochs):
            random.shuffle(train_data)
            for j in range(0, m, k):
                mini_batch_data = train_data[j:j + k]
                self.mini_batch_train(mini_batch_data, alpha)

            if test_data:
                print('Epoch {0}: {1}/{2}'.format(i + 1,
                                                  self.check_test(test_data),
                                                  test_m))
            else:
                print('Epoch {0} finished'.format(i + 1))

    def mini_batch_train(self, train_data, alpha):
        m = len(train_data)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in train_data:
            delta_w, delta_b  = self.BP(x, y)
            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]

        am = alpha / m
        self.weights = [w - am * dw for w, dw in zip(self.weights, nabla_w)]
        self.biases = [b - am * db for b, db in zip(self.biases, nabla_b)]

    def BP(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        a = x
        cache_a = [a]
        cache_z = []
        for layer in range(self.L - 1):
            z = np.dot(self.weights[layer], a) + self.biases[layer]
            a = self.sigmoid(z)
            cache_a.append(a)
            cache_z.append(z)

        # 输出层
        delta = (a - y) * self.sigmoid_prime(z)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, cache_a[-2].T)

        # 隐藏层
        for l in range(2, self.L):
            z = cache_z[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, cache_a[-l -1].T)

        return (nabla_w, nabla_b)

    def FP(self, a):
        for l in range(self.L - 1):
            z = np.dot(self.weights[l], a) + self.biases[l]
            a = self.sigmoid(z)
        return a

    def check_test(self, test_data):
        results = [(np.argmax(self.FP(x)), y) for x, y in test_data]
        return sum(int(y_hat == y) for y_hat, y in results)

