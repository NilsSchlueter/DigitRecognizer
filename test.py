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
    {"input": [1, 0.1, 0.5, 0.5],
     "output": [1, 0.99]},
    {"input": [0.1, 0.05, 1, 1],
     "output": [0.99, 0.01]}]

print("Start Learning")
testNN2 = NeuronalNetwork([2, 3, 1],
                          np.array([
                                [0, 0, 0.8, 0.4, 0.3, 0],
                                [0, 0, 0.2, 0.9, 0.5, 0],
                                [0, 0, 0, 0, 0, 0.3],
                                [0, 0, 0, 0, 0, 0.5],
                                [0, 0, 0, 0, 0, 0.9],
                                [0, 0, 0, 0, 0, 0]]),
                          learn_rate=0.01,
                          fnc_activate_type="identity")
                           
trainingData2 = [
    {"input": [1, 1],
     "output": [3]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [2]},
    {"input": [0, 0],
     "output": [0]}]
    
testData2 = [
      {"input": [1, 1],
     "output": [3]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [2]},
    {"input": [0, 0],
     "output": [0]},
]
  

testNN2.train(trainingData2, 10000)
testNN2.test(trainingData2)
