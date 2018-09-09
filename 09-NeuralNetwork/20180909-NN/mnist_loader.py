# -*- coding: utf-8 -*-
import pickle
import gzip
import numpy as np

def load_data():
    f = gzip.open('../mnist.pkl.gz', 'rb')
    train_data, val_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return train_data, val_data, test_data

def load_data_wrapper():
    train_data, val_data, test_data = load_data()
    train_X = [x.reshape(784, 1) for x in train_data[0]]
    train_y = [vectorize(y) for y in train_data[1]]

    test_X = [x.reshape(784, 1) for x in test_data[0]]
    val_X = [x.reshape(784, 1) for x in val_data[0]]

    return (list(zip(train_X, train_y)),
        list(zip(val_X, val_data[1])),
        list(zip(test_X, test_data[1]))
    )

def vectorize(i):
    y = np.zeros((10, 1))
    y[i] = 1.0
    return y

