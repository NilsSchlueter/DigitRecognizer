from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import numpy as np

trainingData = [
    {"input": [1, 1],
     "output": [0]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]}]

testData = [
    {"input": [1, 1],
     "output": [0]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]}]

networkData = [
    {"fnc_activate_type": "log",
     "learn_rate": 0.9}
]

xorNetwork = NeuronalNetwork(
    layers=[2, 3, 1],
    learn_rate=0.9,
    fnc_activate_type="log"
)

xorNetwork.train(training_data=trainingData, max_iterations=10000)
xorNetwork.test(test_data=testData)
