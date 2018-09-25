# -*- coding: utf-8 -*-

# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data ():
    """
        返回: (train_data, val_date, test_data)
    """
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return (train_data, val_data, test_data)

def load_data_wrapper ():
    """
        由于进行数字认识是一个多分类(multi-classifier)问题，
        所以可以对训练数据进行下预处理:
             1. 使样本的特征的 shape 从 (784,) 变成 (784, 1)
             2. 使分类标签变量变成 (10, 1) , 仅针对训练数据。
    """
    train_data, val_data, test_data = load_data()
    train_X = [x.reshape(784, 1) for x in train_data[0]]
    train_y = [vectorize(y) for y in train_data[1]]

    val_X = [x.reshape(784, 1) for x in val_data[0]]
    test_X = [x.reshape(784, 1) for x in test_data[0]]

    return (
        list(zip(train_X, train_y)),
        list(zip(val_X, val_data[1])),
        list(zip(test_X, test_data[1]))
    )

def vectorize(j):
    v = np.zeros((10, 1))
    v[j] = 1.0
    return v
