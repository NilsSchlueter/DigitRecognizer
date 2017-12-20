from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import sys

class NetworkComparer:

    def __init__(self, networkData, trainingData, testData):
        self.networkData = networkData
        self.trainingData = trainingData
        self.testData = testData

    def compareNetworks(self):

        f = open("test_results.txt", 'w')

        for i in range(len(self.networkData)):

            data = self.networkData[i]

            for j in range(len(data["learn_rate"])):
                curLearnRate = data["learn_rate"][j]
                f.write("\n\n===Using %s with a learn rate of %f ===\n" % (data["fnc_activate_type"], curLearnRate))

                curNetwork = NeuronalNetwork(
                    layers=data["layers"],
                    learn_rate=curLearnRate,
                    fnc_activate_type=data["fnc_activate_type"],
                    fnc_learn_type=data["fnc_learn_type"],
                    weight_matrix=data["weight_matrix"],
                    rnd_values_low=data["rnd_values_low"],
                    rnd_values_high=data["rnd_values_high"])
                #curNetwork.train(training_data=self.trainingData, max_iterations=data["max_iterations"])
                testResults = curNetwork.test(test_data=self.testData)
                f.write(testResults)

        f.close()

