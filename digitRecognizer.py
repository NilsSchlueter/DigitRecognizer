import numpy as np
from networkComparison import NetworkComparer
from neuronalNetwork.neuronalNetwork import NeuronalNetwork
from helpers.csvImporter import CSVImporter

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:1000]
testData1 = trainingData[2000:3500]

# create network
print("Network created!")
#weight_matrix = np.load("weight_matrix_ers_40.npy")
#print(weight_matrix.shape)


network2 = NeuronalNetwork(
    layers=[784, 40, 10],
    fnc_activate_type="tanH",
    learn_rate=0.04,
    fnc_learn_type="ERS",
    rnd_values_low=-1,
    rnd_values_high=1,
    number_epochs=5
)
network2.train(training_data=testData, max_iterations=100)
network2.test(test_data=testData)
