from networkComparison import NetworkComparer
from neuronalNetwork.dynamicNeuronalNetwork import DynamicNeuronalNetwork
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
     "output": [0]},
]

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
     "fnc_learn_type": "BP",
     "learn_rate": [0.9],
     "layers": [2, 3, 3, 1],
     "weight_matrix": None,
     "rnd_values_low": -0.3,
     "rnd_values_high": 0.3,
     "max_iterations": 10000
     }]

#weight_matrix = np.load("weight_matrix_final_np.npy")



weight_matrix_wrong = np.array([
    [0,          0,         -0.09125215,  0.10867563, -0.87053027,  0],
    [0,          0,         -0.28703041, -0.91735518,  0.27542424,  0],
    [0,          0,          0,          0,          0,         -0.22570249],
    [0,          0,          0,          0,          0,          0.70267074],
    [0,          0,          0,          0,          0,         -0.14212258],
    [0,          0,          0,          0,          0,          0]])

weight_matrix_good = np.array([
    [0,          0,         -0.94637122, 0.70390971, 0.20067866,    0],
    [0,          0,         -0.19663551, -0.02842347, -0.50950063,  0],
    [0,          0,          0,          0,          0,         -0.87011895],
    [0,          0,          0,          0,          0,         -0.46769462],
    [0,          0,          0,          0,          0,          0.53744466],
    [0,          0,          0,          0,          0,          0]]
)


network = DynamicNeuronalNetwork(
    layers=[2, 3, 3, 1],
    fnc_learn_type="BP",
    fnc_activate_type="log",
    learn_rate=0.9,
    rnd_values_low=-1,
    rnd_values_high=1,
    weight_matrix=None
)



#networkComparer = NetworkComparer(networkData=networkData, trainingData=trainingData, testData=testData)
#networkComparer.compareNetworks()
network.train(training_data=trainingData, max_iterations=10000)
network.test(test_data=testData)
