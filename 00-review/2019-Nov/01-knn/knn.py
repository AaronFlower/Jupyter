import numpy as np
import matplotlib.pyplot as plt

def kNNClassifier(X, y, k, inX):
    dist = np.sum((X - inX) ** 2, axis=1) ** 0.5
    sortIndices = dist.argsort() # 下标升序

    countDict = {}
    retLabel = None
    maxCount = 0
    for i in range(k):
        label = y[sortIndices[i]]
        countDict[label] = countDict.get(label, 0) + 1
        if countDict[label] >= maxCount:
            retLabel = label
            maxCount = countDict[label]

    print(countDict)
    return retLabel

if __name__ == "__main__":
    X = np.array([[1,2],[1,1.5],[3,2],[3,3]])
    y = np.array([1, 1, 2, 2])
    l = kNNClassifier(X, y, 3, [1.5, 1.2])
    print(l)
