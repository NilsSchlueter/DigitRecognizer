from neuronalNetwork.NeuralNetwork import NeuralNetwork
from helpers.csvImporter import CSVImporter
from helpers.networkComparison import NetworkComparer
import numpy as np

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_file("ressources/train.csv")
testData = csvImporter.import_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:100]
testData1 = trainingData[2000:3500]

# create network
print("Network created!")
# weight_matrix = np.load("weight_matrix_ers_40.npy")
# print(weight_matrix.shape)

'''
data = {
    "layers": [784, 20, 10],
    "fnc_activate_type": "log",
    "fnc_learn_type": "ERS",
    "rnd_values_low": -1,
    "rnd_values_high": 1,
    "number_epochs": 2,
    "weight_matrix": None,
    "learn_rate": [0.04, 0.02],
    "num_hidden": [40, 60, 80, 100],
    "epochs": 1
}

comparer = NetworkComparer(data, trainingData1, testData)
comparer.compare_networks()

'''
# m = np.load("ers2_100_log_85perc.npy")

network2 = NeuralNetwork(
    layers=[784, 60, 10],
    fnc_activate_type="log",
    learn_rate=0.04,
    fnc_learn_type="ERS",
    rnd_values_low=-1,
    rnd_values_high=1,
    number_epochs=1,
    weight_matrix=None
)
network2.train(training_data=trainingData)
network2.test(testData, print_results=True)
