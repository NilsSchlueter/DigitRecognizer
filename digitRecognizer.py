import numpy as np
from csvReader.csvImporter import CSVImporter
from neuronalNetwork.neuronalNetwork import NeuronalNetwork

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_test_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:-50]
testData1 = trainingData[-50:]

# create network
print("Creating the nework...")
network = NeuronalNetwork(
    layers=[1784, 15, 9],
    fnc_activate_type="log",
    learn_rate=0.01
)
print("Network created!")

print("Training the network...")
network.train(trainingData1, 1000)
network.test(testData1)
