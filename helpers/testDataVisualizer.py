import numpy as np
from matplotlib import pyplot as plt

class Visualizer:

    def __init__(self):
        pass

    def visualize(self, data, row_length=28):

        imgData = np.array(data).reshape((row_length, row_length))
        plt.imshow(imgData, cmap=plt.cm.Greys)
        plt.show()



visualizer = Visualizer()
