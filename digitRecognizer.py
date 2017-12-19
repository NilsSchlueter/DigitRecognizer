import numpy as np
from networkComparison import NetworkComparer
from csvReader.csvImporter import CSVImporter
from neuronalNetwork.neuronalNetwork import NeuronalNetwork


# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:-10000]
testData1 = trainingData[-100:]

# create network
print("Network created!")

weight_matrix = np.load("weight_matrix_final_np.npy")

networkData = [{
        "layers": [784, 20, 20, 10],
        "fnc_activate_type": "log",
        "fnc_learn_type": "BP",
        "learn_rate": [0.9],
        "weight_matrix": None, # weight_matrix,
        "rnd_values_low": -1,
        "rnd_values_high": 1,
        "max_iterations": 40000
    }]

networkComparer = NetworkComparer(
    networkData=networkData,
    trainingData=trainingData1,
    testData=testData1
)


print("Train network!")
networkComparer.compareNetworks()
print("Network trained!")
