import numpy as np
from networkComparison import NetworkComparer
from neuronalNetwork.dynamicNeuronalNetwork import DynamicNeuronalNetwork
from helpers.csvImporter import CSVImporter

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:-1000]
testData1 = trainingData[-1000:]

# create network
print("Network created!")
#weight_matrix = np.load("weight_matrix_BP_with_8000_Iterations.npy")
#print(weight_matrix.shape)


networkData = [{
        "layers": [784, 20, 10],
        "fnc_activate_type": "log",
        "fnc_learn_type": "BP",
        "learn_rate": [0.3],
        "weight_matrix": None,
        "rnd_values_low": -0.5,
        "rnd_values_high": 0.5,
        "max_iterations": 80000
    }]

networkComparer = NetworkComparer(
    networkData=networkData,
    trainingData=trainingData,
    testData=testData
)


print("Train network!")
networkComparer.compareNetworks()
print("Network trained!")

'''
network2 = DynamicNeuronalNetwork(
    layers=[784, 15, 15, 10],
    fnc_activate_type="log",
    learn_rate=0.9,
    fnc_learn_type="BP",
    rnd_values_low=-1,
    rnd_values_high=1
)
network2.train(training_data=trainingData1, max_iterations=40000)
network2.test(test_data=testData1)
'''