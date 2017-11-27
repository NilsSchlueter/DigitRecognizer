import numpy as np
from networkComparison import NetworkComparer
from csvReader.csvImporter import CSVImporter
from neuronalNetwork.neuronalNetwork import NeuronalNetwork


# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:3]
testData1 = trainingData[:3]

# create network
print("Network created!")
networkData = [{
        "layers": [1784, 20, 10],
        "fnc_activate_type": "log",
        "learn_rate": [0.9],
        "max_iterations": 100
    }]

networkComparer = NetworkComparer(
    networkData=networkData,
    trainingData=trainingData,
    testData=testData
)

print("Train network!")
networkComparer.compareNetworks()
print("Network trained!")
