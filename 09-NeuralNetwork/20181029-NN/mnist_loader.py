# -*- coding: utf-8 -*-

import gzip
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def load_data_wrapper ():
    f = gzip.open('../mnist.pkl.gz', 'rb')
    train, val, test = pickle.load(f, encoding = 'bytes')
    f.close()

    train_X = [x.reshape(-1, 1) for x in train[0]]
    train_y = [vectorize(y) for y in train[1]]

    val_X = [x.reshape(-1, 1) for x in val[0]]
    test_X = [x.reshape(-1, 1) for x in test[0]]

    return (
        list(zip(train_X, train_y)),
        list(zip(val_X, val[1])),
        list(zip(test_X, test[1]))
    )


def vectorize (i):
    y = np.zeros((10, 1))
    y[i] = 1
    return y

def plot_images6 (data):
    ilist = np.random.permutation(len(data))
    fig = plt.figure()

    for i in range(1, 7):
        X, _ = data[ilist[i]]
        image = X.reshape(28, 28)
        ax = fig.add_subplot(1, 6, i)
        ax.matshow(image, cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

    return


