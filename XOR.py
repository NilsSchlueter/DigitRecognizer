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

network = NeuronalNetwork(

    layers=[2, 3, 1],
    fnc_learn_type="BP",
    fnc_activate_type="log",
    weight_matrix=None,
    learn_rate=0.9,

    rnd_values_low=-1.0,
    rnd_values_high=1.0
)

network.train(training_data=trainingData, max_iterations=1000)
network.test(test_data=testData)
'''