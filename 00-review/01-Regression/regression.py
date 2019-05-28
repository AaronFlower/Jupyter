import numpy as np
import matplotlib.pyplot as plt

def loadData (filename):
    data = np.loadtxt(filename)
    features = data[:, :-1]
    labels = data[:, -1]
    X = np.mat(features)
    y = np.mat(labels[:, np.newaxis])
    return X, y

def lseWeights(X, y):
    '''
    最小二乘法
    '''
    return (X.T * X).I * X.T * y

def computeCostJ(X, y, ws):
    m,_ = X.shape
    diff = X * ws - y
    cost = diff.T * diff / (2 * m)
    return np.asarray(cost).flatten()[0]

def lrGradientDescent(X, y, alpha = 0.1, iter_nums = 100):
    m, n = X.shape
    ws = np.zeros((n, 1))
    J_history = []
    J_history.append(computeCostJ(X, y, ws))

    for i in range(iter_nums):
        h = X * ws
        # ws = ws - alpha * X.T * (h - y)
        # 另忘记了分母 m, 自己可以推导出来是有分母的
        ws = ws - (alpha / m)  * X.T * (h - y)
        J_history.append(computeCostJ(X, y, ws))

    return ws, J_history


def plotData(X, y, ws):
    # Plot the samples
    xPos = X.getA()[:,1]
    yPos = y.getA()[:,0]
    plt.figure(figsize=(20,7))
    plt.scatter(xPos, yPos)

    # Plot the regression line
    xPos = np.arange(0, 1, 0.01)
    yPos = np.squeeze(np.asarray(ws[0] + ws[1] * xPos ))
    plt.plot(xPos, yPos)

    plt.show()

def run():
    X, y = loadData('ex1.txt')
    ws, his = lrGradientDescent(X, y)
    fig, axes = plt.subplots(2, 1)

    xPos = X.getA()[:, 1]
    yPos = y.getA()[:, 0]
    axes[0].scatter(xPos, yPos)

    xPos = np.arange(0, 1, 0.01)
    yPos = np.squeeze(np.asarray(ws[0] + ws[1] * xPos))
    axes[0].plot(xPos, yPos)

    xPos = np.arange(len(his))
    axes[1].plot(xPos, his)
