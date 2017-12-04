from neuronalNetwork.neuronalNetwork import NeuronalNetwork
import sys

class NetworkComparer:

    def __init__(self, networkData, trainingData, testData):
        self.networkData = networkData
        self.trainingData = trainingData
        self.testData = testData

    def compareNetworks(self):

        # Redirect print to file
        orig_stdout = sys.stdout
        f = open("result.txt", 'w')
        sys.stdout = f

        for i in range(len(self.networkData)):

            data = self.networkData[i]

            for j in range(len(data["learn_rate"])):
                curLearnRate = data["learn_rate"][j]
                print("\n\n===Using %s with a learn rate of %f ===\n" % (data["fnc_activate_type"], curLearnRate))

                curNetwork = NeuronalNetwork(
                    layers=data["layers"],
                    learn_rate=curLearnRate,
                    fnc_activate_type=data["fnc_activate_type"])
                curNetwork.train(training_data=self.trainingData, max_iterations=data["max_iterations"])
                curNetwork.test(test_data=self.testData)


        sys.stdout = orig_stdout
        f.close()

