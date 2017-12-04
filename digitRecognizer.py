import numpy as np
from networkComparison import NetworkComparer
from csvReader.csvImporter import CSVImporter
from neuronalNetwork.neuronalNetwork import NeuronalNetwork


# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:5]
testData1 = trainingData[:5]

# create network
print("Network created!")

weight_matrix = np.load("weight_matrix_final_np.npy")

networkData = [{
        "layers": [784, 20, 10],
        "fnc_activate_type": "log",
        "learn_rate": [0.9],
        "weight_matrix": None, #weight_matrix,
        "rnd_values_low": -1.0,
        "rnd_values_high": 1.0,
        "max_iterations": 300
    }]

networkComparer = NetworkComparer(
    networkData=networkData,
    trainingData=trainingData1,
    testData=testData1
)


print("Train network!")
networkComparer.compareNetworks()
print("Network trained!")
