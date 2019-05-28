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
