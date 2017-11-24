import numpy as np
from csvReader.csvImporter import CSVImporter
from neuronalNetwork.neuronalNetwork import NeuronalNetwork

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:50]
testData1 = testData

# create network
print("Creating the nework...")
network = NeuronalNetwork(
    layers=[1784, 15, 10],
    fnc_activate_type="log",
    learn_rate=0.9
)
print("Network created!")

print("Training the network...")
network.train(trainingData1, 1500)
print("Test the network")
network.test(testData1)
