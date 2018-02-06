from neuronalNetwork.NeuralNetwork import NeuralNetwork
import numpy as np

trainingData = [
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [0, 0, 0]]

testData = [
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
    [0, 0, 0]]

# m = np.load("neuneuneu.npy")

network = NeuralNetwork(
    layers=[2, 3, 3, 1],
    fnc_learn_type="BP",
    fnc_activate_type="log",
    weight_matrix=None,
    learn_rate=0.7,
    rnd_values_low=-1,
    rnd_values_high=1,
    number_epochs=10000
)

network.train(training_data=trainingData)
network.test(testData, True)
