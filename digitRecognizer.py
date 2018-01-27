from neuronalNetwork.neuronalNetwork2 import NeuronalNetwork
from helpers.csvImporter import CSVImporter

# Import data from the csv files
csvImporter = CSVImporter()
trainingData = csvImporter.import_file("ressources/train.csv")
testData = csvImporter.import_file("ressources/test.csv")

# Create simple training and test data
trainingData1 = trainingData[:1000]
testData1 = trainingData[2000:3500]

# create network
print("Network created!")
# weight_matrix = np.load("weight_matrix_ers_40.npy")
#print(weight_matrix.shape)


network2 = NeuronalNetwork(
    layers=[784, 20, 10],
    fnc_activate_type="log",
    learn_rate=0.3,
    fnc_learn_type="BP",
    rnd_values_low=-1,
    rnd_values_high=1,
    number_epochs=2,
    weight_matrix=None
)
network2.train(training_data=trainingData)
