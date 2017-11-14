from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import numpy as np

testNN = NeuronalNetwork([4, 2, 2],
                         np.array([
                                 [0,0, 0, 0, 0.15, 0.25, 0, 0],
                                 [0,0, 0, 0, 0.2, 0.3, 0, 0],
                                 [0,0, 0, 0, 0.35, 0.35, 0, 0],
                                 [0,0, 0, 0, 0, 0, 0.6, 0.6],
                                 [0,0, 0, 0, 0, 0, 0.4, 0.5],
                                 [0,0, 0, 0, 0, 0, 0.45, 0.55],
                                 [0,0, 0, 0, 0, 0, 0, 0],
                                 [0,0, 0, 0, 0, 0, 0, 0]]))
trainingData = [
    {"input": [0.05, 0.1, 1, 1],
     "output": [0.01, 0.99]}]

testNN.train(trainingData)
"""
testNN2 = NeuronalNetwork([3,1],
                           np.array([
                                [0, 0, 0, 0.6],
                                [0, 0, 0, 0.6],
                                [0, 0, 0, 0.6],
                                [0, 0, 0, 0]]))
                           
trainingData2 = [
    {"input": [1, 1, 1],
     "output": 1},
    {"input": [1, 1, 0],
     "output": 1},
    {"input": [1, 0, 1],
     "output": 1},
    {"input": [0, 1, 1],
     "output": 1},
    {"input": [1, 0, 0],
     "output": 0},
    {"input": [0, 1, 0],
     "output": 0},
    {"input": [0, 0, 1],
     "output": 0},
    {"input": [0, 0, 0],
     "output": 0},
]
    
testNN2.train(trainingData2);

"""
