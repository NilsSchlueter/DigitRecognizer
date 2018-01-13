import numpy as np
from networkComparison import NetworkComparer
from neuronalNetwork.neuronalNetwork import NeuronalNetwork
from helpers.csvImporter import CSVImporter

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_training_file("ressources/train.csv")
testData = csvImporter.import_training_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:100]
testData1 = trainingData[2000:3500]

# create network
print("Network created!")
weight_matrix = np.load("weight_matrix_final_np.npy")
#print(weight_matrix.shape)



network2 = NeuronalNetwork(
    layers=[784, 20, 10],
    fnc_activate_type="log",
    learn_rate=0.1,
    weight_matrix=weight_matrix,
    fnc_learn_type="BP",
    rnd_values_low=-1.0,
    rnd_values_high=1.0
)
network2.train(training_data=trainingData, max_iterations=40000)
network2.test(test_data=testData1)
