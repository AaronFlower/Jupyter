# encoding: utf-8

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def randomImageShow(data):
    indexList = np.random.permutation(len(data))

    _, axes = plt.subplots(1, 6)

    for i in range(6):
        image, _ = data[indexList[i]]
        axes[i].imshow(image.reshape(28, 28), cmap=matplotlib.cm.binary)
    plt.show()
    return
