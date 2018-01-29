from neuronalNetwork.NeuralNetwork import NeuralNetwork


class NetworkComparer:
    def __init__(self, network_data, training_data, test_data):
        self.networkData = network_data
        self.trainingData = training_data
        self.testData = test_data

    def compare_networks(self):

        data = self.networkData
        f = open("comparison_results.txt", "w")

        for j in range(len(data["learn_rate"])):
            for k in range(len(data["num_hidden"])):
                cur_learn_rate = data["learn_rate"][j]
                cur_hidden_neurons = data["num_hidden"][k]
                cur_layers = data["layers"]
                cur_layers[1] = cur_hidden_neurons

                print("Now testing with learn rate: %s and layers: %s\n" % (cur_learn_rate, cur_layers))

                cur_network = NeuralNetwork(
                    layers=cur_layers,
                    learn_rate=cur_learn_rate,
                    fnc_activate_type=data["fnc_activate_type"],
                    fnc_learn_type=data["fnc_learn_type"],
                    weight_matrix=data["weight_matrix"],
                    rnd_values_low=data["rnd_values_low"],
                    rnd_values_high=data["rnd_values_high"],
                    number_epochs=data["epochs"])

                cur_network.train(training_data=self.trainingData)
                result, result_str = cur_network.test(self.testData, True)
                f.write(result_str)

        f.close()