import numpy as np
from csvReader.csvImporter import CSVImporter
from neuronalNetwork.neuronalNetwork import NeuronalNetwork

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:15]
testData1 = trainingData[:15]

# create network
print("Creating the nework...")
network = NeuronalNetwork(
    layers=[1784, 20, 10],
    fnc_activate_type="log",
    learn_rate=0.5
)
print("Network created!")

print("Training the network...")
network.train(trainingData1, 300)
print("Test the network")
network.test(testData1)
