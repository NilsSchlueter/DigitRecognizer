from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import numpy as np

testNN = NeuronalNetwork([4, 2, 2],
                         np.array([
                                 [0, 0, 0, 0, 0.2, 0.3, 0, 0],
                                 [0, 0, 0, 0, 0.1, 0.4, 0, 0],
                                 [0, 0, 0, 0, 0.2, 0.3, 0, 0],
                                 [0, 0, 0, 0, 0.2, 0.5, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0.5, 0.3],
                                 [0, 0, 0, 0, 0, 0, 0.2, 0.5],
                                 [0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0]]))
trainingData = [
    {"input": [1, 0, 0, 0],
     "output": [0, 0]},

    {"input": [0, 1, 0, 0],
     "output": [1, 0]},

    {"input": [0, 0, 1, 0],
     "output": [0, 1]},

    {"input": [0, 0, 0, 1],
     "output": [1, 1]},
]

testNN.train(trainingData)

