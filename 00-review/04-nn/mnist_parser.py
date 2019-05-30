# encoding: utf-8

import pickle
import gzip

import numpy as np

def parse():
    '''
        解析 minist.pkl.gz 数据，返回训练数据，验证数据和测试数据
        每个样本的特征数为：784, 分类的: 0 ~ 9
        train_data: 50000 的样本集，
        val_data: 10000 的样本集，
        test_data: 10000 的样本集，
    '''
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_tuple, val_tuple, test_tuple = pickle.load(f, encoding='bytes')

    train = reshape(train_tuple)
    val = reshape(val_tuple)
    test= reshape(test_tuple)

    return train, val, test

def reshape(data):
    '''
        将样本的 (784,) 转换成 (784, 1)
        将分类转换成 （10,1) 的分类向量
    '''
    data_x, data_y = data
    x = [x.reshape(784, 1) for x in data_x]
    y = [vectorize_y(y) for y in data_y]
    return list(zip(x, y))

def vectorize_y(y):
    '''
        0 ~ 9: 10 个分类
    '''
    v = np.zeros((10, 1))
    v[y] = 1
    return v
