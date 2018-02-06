from neuronalNetwork.NeuralNetwork import NeuralNetwork
from helpers.csvImporter import CSVImporter

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_file("ressources/train.csv")
testData = csvImporter.import_file("ressources/test.csv")

# create network
print("Network created!")
network = NeuralNetwork(
    layers=[784, 80, 10],
    fnc_activate_type="log",
    learn_rate=0.04,
    fnc_learn_type="BP",
    rnd_values_low=-1,
    rnd_values_high=1,
    number_epochs=1,
    weight_matrix=None
)
network.train(training_data=trainingData)
network.test(testData, print_results=True)
