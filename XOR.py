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
    layers=[2, 100, 1],
    fnc_learn_type="ERS",
    fnc_activate_type="log",
    weight_matrix=None,
    learn_rate=0.08,
    rnd_values_low=-0.3,
    rnd_values_high=0.3,
    number_epochs=10000
)

network.train(training_data=trainingData, max_iterations=1000)
network.test(test_data=testData)
