'''
Logistic Regression
Notes: 1. cost function; 2. predict, 3. plot decision boundary
'''
import numpy as np
import matplotlib.pyplot as plt

def loadData(filename = 'testSet01.txt'):
    data = np.loadtxt(filename)
    X = data[:,0:-1]
    X = np.insert(X, 0, 1.0, axis=1)
    y = data[:, -1][:, np.newaxis]
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X, y, theta):
    m = len(y)
    yHat = sigmoid(np.dot(X, theta))
    class1_cost = -y * np.log(yHat)
    class2_cost = (1 - y) * np.log(1 - yHat)

    cost = class1_cost + class2_cost;

    return cost.sum() / m


def logisticRegression(X, y, alpha = 0.1, num_iters = 100):
    m, n = X.shape
    his = []
    theta = np.zeros((n, 1))
    xT = X.transpose()

    his.append(computeCost(X, y, theta))
    for _ in range(num_iters):
        # his.append(computeCost(X, y, theta))
        yHat = sigmoid(np.dot(X, theta))
        theta = theta - (alpha / m) * np.dot(xT, (yHat - y))
        his.append(computeCost(X, y, theta))
    return theta, his

def run():
    X, y = loadData()
    theta, his = logisticRegression(X, y, 0.01, 1000)


    posIdx = y == 1
    negIdx = y == 0

    posX = X[posIdx.flatten()]
    negX = X[negIdx.flatten()]

    fig, axes = plt.subplots(2, 1)
    axes[0].scatter(posX[:, 1], posX[:, 2], marker='x')
    axes[0].scatter(negX[:, 1], negX[:, 2], marker='o')

    xPos = np.arange(-3.5, 3.5, 0.1)
    yPos0 = (- theta[0] - theta[1] * xPos)/theta[2]
    yPos1 = (1 - theta[0] - theta[1] * xPos)/theta[2]
    axes[0].plot(xPos, yPos0, 'b-')
    axes[0].plot(xPos, yPos1, 'r.')

    xPos = np.arange(len(his))
    axes[1].plot(xPos, his)

    return
