import numpy as np
from matplotlib import pyplot as plt


def visualize(data, row_length=28):

    if isinstance(data, list):

        for entry in data:
            imgData = np.array(entry["input"]).reshape((row_length, row_length))
            plt.imshow(imgData, cmap=plt.cm.Greys)
        plt.show()

    else:
        imgData = np.array(data).reshape((row_length, row_length))
        plt.imshow(imgData, cmap=plt.cm.Greys)
        plt.show()



