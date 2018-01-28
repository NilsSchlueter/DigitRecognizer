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


data = {
    "layers": [784, 20, 10],
    "fnc_activate_type": "log",
    "fnc_learn_type": "BP",
    "rnd_values_low": -1,
    "rnd_values_high": 1,
    "number_epochs": 2,
    "weight_matrix": None,
    "learn_rate": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "num_hidden": [20, 40, 60],
    "epochs": 1
}

#comparer = NetworkComparer(data, trainingData, testData)
#comparer.compare_networks()

# m = np.load("weight_matrix_final.npy")

network2 = NeuralNetwork(
    layers=[784, 800, 10],
    fnc_activate_type="log",
    learn_rate=0.9,
    fnc_learn_type="BP",
    rnd_values_low=-1,
    rnd_values_high=1,
    number_epochs=5,
    weight_matrix=None
)
network2.train(training_data=trainingData1)
network2.test(trainingData1, print_results=True)
