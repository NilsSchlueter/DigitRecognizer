from networkComparison import NetworkComparer
from neuronalNetwork.dynamicNeuronalNetwork import DynamicNeuronalNetwork
from neuronalNetwork.neuronalNetwork import NeuronalNetwork

trainingData = [
    {"input": [1, 1],
     "output": [1]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]}]

testData = [
    {"input": [1, 1],
     "output": [1]},
    {"input": [0, 1],
     "output": [1]},
    {"input": [1, 0],
     "output": [1]},
    {"input": [0, 0],
     "output": [0]}]

networkData = [
    {"fnc_activate_type": "log",
     "fnc_learn_type": "BP",
     "learn_rate": [0.9],
     "layers": [2, 3, 1],
     "weight_matrix": None,
     "rnd_values_low": -0.3,
     "rnd_values_high": 0.3,
     "max_iterations": 10000,
     "epochs": 10
     }]

#weight_matrix = np.load("weight_matrix_80_percent.npy")

networkComparer = NetworkComparer(networkData=networkData, trainingData=trainingData, testData=testData)
networkComparer.compareNetworks()

'''
network = DynamicNeuronalNetwork(
    layers=[2, 3, 1],
    fnc_learn_type="BP",
    fnc_activate_type="log",
    weight_matrix=None,
    learn_rate=0.9,
    rnd_values_low=-1,
    rnd_values_high=1
)

network.train(training_data=trainingData, max_iterations=100000)
network.test(test_data=testData)
'''