from networkComparison import NetworkComparer
from neuronalNetwork.dynamicNeuronalNetwork import DynamicNeuronalNetwork

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
    {"fnc_activate_type": "tanH",
     "fnc_learn_type": "BP",
     "learn_rate": [0.1],
     "layers": [2, 3, 2, 1],
     "weight_matrix": None,
     "rnd_values_low": -0.3,
     "rnd_values_high": 0.3,
     "max_iterations": 1
     }]

#weight_matrix = np.load("weight_matrix_final_np.npy")

network = DynamicNeuronalNetwork(
    layers=[2, 3, 2, 1],
    weight_matrix=None,
    learn_rate=0.9,
    rnd_values_low=-1,
    rnd_values_high=1
)

