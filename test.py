from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import numpy as np

testNN = NeuronalNetwork([3, 2, 3])
print(testNN.weight_matrix)
print(testNN.fnc_propagate_type)
print(testNN.fnc_activate_type)
print(testNN.fnc_output_type)