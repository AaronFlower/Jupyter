# encoding: utf-8

'''
实现随机梯度下降的 NN
'''

import random
import numpy as np

def sigmoid(z):
    '''
        compute the sigmoid function.
    '''
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    '''
        Derivative of the sigmoid function
    '''
    f = sigmoid(z)
    return f * (1 - f)

class Network(object):

    def __init__(self, layer_sizes):
        '''
            初始化 Network, 初始化每层的权重矩阵，偏置向量.
            - layer_sizes : array-like, 用于指定每层的神经元个数。
            如：layer_sizes = [3, 4, 4, 1], 则说明有 4 层，每层的神经元个数分别
            为：3, 4, 4, 1.
        '''
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        self.biases = [np.random.randn(nLayer, 1) for nLayer in layer_sizes[1:]]
        self.weights = [
            np.random.randn(nLayer, nPreLayer)
            for nLayer, nPreLayer in zip(layer_sizes[1:], layer_sizes[:-1])
        ]

        return

    def FP(self, x):
        '''
            Forward Process, 前向计算
        '''
        a = x
        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            a = sigmoid(z)
        return a

    def train(self,
              train_data,
              epoches,
              batch_size,
              alpha,
              test_data = None):
        '''
            使用 mini-batch stochastic gradient descent 来训练模型
            - train_data : 训练数据集
            - epoches: 迭代次数
            - batch_size: 更新梯度时每次所用的样本数
            - alpha: 学习因子
            - test_data: 测试数据集
        '''
        num_train = len(train_data)
        if test_data:
            num_test = len(test_data)

        for i in range(epoches):
            random.shuffle(train_data)
            batches = [
                train_data[k : k + batch_size]
                for k in range(0, num_train, batch_size)
            ]

            for batch in batches:
                self.batch_GD(batch, alpha)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), num_test
                ))
            else:
                print("Epock {0} complete".format(i))

        return

    def batch_GD(self, train_data, alpha):
        '''
            批量梯度下降, 需要用 BP 来计算梯度
            - train_data: 训练数据集
            - alpha: 学习因子
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 对数据集中的每个样本的进行累加
        for x, y in train_data:
            delta_nabla_b, delta_nabla_w = self.BP(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        am = alpha / len(train_data)
        self.weights = [w - am * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - am * nb for b, nb in zip(self.biases, nabla_b)]

        return

    def BP(self, x, y):
        '''
            Backward Process
            - x: 训练样本特征
            - y: 训练样本分类
            Return: (nabla_b, nabla_w), 即 (𝛁b, 𝛁w), 其每层的 b, w 的维度一致.
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # FP

        a = x
        cache_a = [x] # cache all the activations for all layer
        cache_z = []  # cache all the z   vectors for all layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            cache_z.append(z)
            cache_a.append(a)

        # Backward process Output layer delta
        delta = (a - y) * sigmoid_prime(z)

        # 计算输出层的 𝛁b, 𝛁w
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, cache_a[-2].T)

        # 计算隐藏层的 (𝛁b, 𝛁w)
        for i in range(2, self.num_layers):
            z = cache_z[-i]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-i + 1].T, delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, cache_a[-i -1].T)

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        '''
            返回测试样本的分类数
        '''
        test_results = [
            (np.argmax(self.FP(x)), np.argmax(y)) for (x, y) in test_data
        ]

        return sum(int(x == y) for (x, y) in test_results)
