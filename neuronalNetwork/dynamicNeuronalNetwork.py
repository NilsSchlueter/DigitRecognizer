from random import randint
import sys
import numpy as np
import pprint


class DynamicNeuronalNetwork:

    def __init__(self, layers, weight_matrix=None, fnc_propagate_type=None, fnc_activate_type=None, fnc_learn_type="BP",
                 fnc_output_type=None, treshold=0.5, learn_rate=0.5, rnd_values_low=-1.0, rnd_values_high=1.0):

        # Define some global properties
        self.layers = layers
        self.numLayers = len(layers)
        self.numNeurons = sum(layers)
        self.numInputNeurons = layers[0]
        self.numOutputNeurons = layers[self.numLayers - 1]
        self.numHiddenNeurons = sum(layers) - self.numInputNeurons - self.numOutputNeurons
        self.numHiddenLayers = self.numLayers - 2
        self.learnRate = learn_rate
        self.treshold = treshold

        # Function type
        self.fnc_propagate_type = fnc_propagate_type
        self.fnc_activate_type = fnc_activate_type
        self.fnc_output_type = fnc_output_type
        self.fnc_learn_type = fnc_learn_type  # BP, ERS, ERS2

        # Set the functions to default values if no other values are provided
        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        if weight_matrix is None:
            self.__generate_weight_matrix(rnd_values_low=rnd_values_low, rnd_values_high=rnd_values_high)
        else:
            self.weight_matrix = weight_matrix

        self._tempWeightMatrix = self.weight_matrix


        # Generate Neuron information
        self.neurons = []
        num_neurons = 0
        for layerId in range(len(layers)):

            neurons = []
            for i in range(layers[layerId]):
                neuron = {
                    "id": num_neurons,
                    "layer_id": layerId,
                    "layer_type": "input" if layerId == 0 else "output" if layerId == len(layers) - 1 else "hidden",
                    "propagate_value": 0,    # Value of the neuron after the NETTO INPUT function
                    "activation_value": 0,   # Value of the neuron after the ACTIVATION function
                    "output_value": 0,       # Value of the neuron after the OUTPUT function
                }
                neurons.append(neuron)
                num_neurons += 1

            self.neurons.append(neurons)

        self.delta = np.zeros(self.numNeurons)

    def train(self, training_data, max_iterations):

        try:
            f = open("weight_matrix_evolution.txt", 'w')

            iteration = 0
            error_sum = 0
            while iteration < max_iterations:

                iteration += 1

                # Get random training data
                i = randint(0, len(training_data) - 1)

                # Read current input and output vectors
                input_vector = training_data[i]["input"]
                output_vector = training_data[i]["output"]

                # Set input pattern to neurons in first layer
                for j in range(len(input_vector)):
                    self.__set_neuron_value(neuron_id=j, value=input_vector[j], value_type="activation_value")

                increase_error = False

                # Calculate output for the other neurons
                for k in range(self.numInputNeurons, self.numNeurons, 1):
                    self.__set_neuron_value(neuron_id=k, value=self.__fnc_output(k), value_type="activation_value")

                    # Calculate output neurons with treshold
                    if k >= (self.numHiddenNeurons + self.numInputNeurons):
                        output_value = 1 if self.__get_neuron_value(k, "activation_value") > self.treshold else 0
                        self.__set_neuron_value(neuron_id=k, value=output_value, value_type="output_value")

                        if output_value != output_vector[k - self.numHiddenNeurons - self.numInputNeurons]:
                            increase_error = True

                if increase_error:
                    error_sum += 1

                # Change weight matrix
                self.__fnc_learn(output_vector)
                self.weight_matrix = self._tempWeightMatrix

                # Print every 100th weight matrix and error sum to file
                if iteration % 100 == 0:
                    f.write("Weight Matrix after %s Iterations:\n" % (iteration))
                    f.write(str(self.weight_matrix))
                    f.write("\nError Sum: ")
                    f.write(str(error_sum / 100))
                    print("Iteration %s/%s" % (iteration, max_iterations))
                    print("Wrong classifications in the last 100 iterations: " + str(error_sum / 100))
                    f.write("\n\n")

                    error_sum = 0

            f.close()

            f_final = open("weight_matrix_final.txt", "w")
            f_final.write(str(self.weight_matrix))
            f_final.close()

            # Save as numpy
            np.save("weight_matrix_final_np", self.weight_matrix)

        except KeyboardInterrupt:
            f_final = open("weight_matrix_final.txt", "w")
            f_final.write(str(self.weight_matrix))
            f_final.close()

            np.save("weight_matrix_final_np", self.weight_matrix)
            sys.exit()

    def test(self, test_data):

        resultStr = ""
        count = 0
        for i in range(len(test_data)):

            increase_count = True

            input_vector = test_data[i]["input"]
            output_vector = test_data[i]["output"]

            # Set input pattern to neurons in first layer
            for j in range(len(input_vector)):
                self.__set_neuron_value(neuron_id=j, value=input_vector[j], value_type="activation_value")

            # Calculate output for the other neurons
            for k in range(self.numInputNeurons, self.numNeurons, 1):
                self.__set_neuron_value(neuron_id=k, value=self.__fnc_output(k), value_type="activation_value")

                # Calculate output neurons with treshold
                if k >= (self.numHiddenNeurons + self.numInputNeurons):
                    output_value = 1 if self.__get_neuron_value(k, "activation_value") > self.treshold else 0
                    self.__set_neuron_value(neuron_id=k, value=output_value, value_type="output_value")

            out = np.zeros(self.numOutputNeurons)
            for l in range(self.numOutputNeurons):
                out[l] = self.__get_neuron_value(neuron_id=l + self.numInputNeurons + self.numHiddenNeurons, value_type="activation_value")
                if self.__get_neuron_value(l + self.numInputNeurons + self.numHiddenNeurons, "output_value") != output_vector[l]:
                    increase_count = False

            if increase_count:
                count += 1

            resultStr += "Outputvector: %s Targetvector: %s Resultvector: %s\n" % (out, output_vector, self.neurons)

        percent = count / len(test_data)
        resultStr += "%s wurden erfolgreich erkannt" % (percent)

        f = open("test_results.txt", 'w')
        f.write(resultStr)
        f.close()

        return resultStr

    def __fnc_learn(self, output_vector):

        error_sum = 0

        # Loop through every layer except for the input layer
        for i in range(self.numLayers - 1, 0, -1):

            # Loop through each neuron in the layer
            for j in range(len(self.neurons[i])):
                cur_neuron = self.neurons[i][j]

                # Get target and actual value
                actual_value = cur_neuron["activation_value"]
                real_output = cur_neuron["output_value"]
                error = 0

                # If this is an output neuron, get the target value to calculate the error
                if cur_neuron["layer_type"] == "output":
                    target_value = output_vector[j]

                    # Calculate error
                    error = self.__calculate_error(target_value, real_output)
                    error_sum += error


                # If this is a hidden neuron
                elif cur_neuron["layer_type"] == "hidden":

                    weight_row = self.weight_matrix[cur_neuron["id"], :]
                    for n in range(len(weight_row)):
                        if weight_row[n] != 0:
                            error += self.delta[n] * weight_row[n]

                # Delta Calculation depending on learn function
                if self.fnc_learn_type == "BP":
                    derivative_value = self.__derivative_activation(actual_value)
                    self.delta[cur_neuron["id"]] = error * derivative_value
                elif self.fnc_learn_type == "ERS" or self.fnc_learn_type == "ERS2":
                    self.delta[cur_neuron["id"]] = error

                # Loop through each neuron in previous layer
                for k in range(len(self.neurons[i - 1])):
                    prev_neuron = self.neurons[i-1][k]

                    if self.fnc_learn_type == "BP":
                        self._tempWeightMatrix[prev_neuron["id"]][cur_neuron["id"]] = self.weight_matrix[prev_neuron["id"]][cur_neuron["id"]] - self.learnRate * self.delta[cur_neuron["id"]] * cur_neuron["activation_value"]

                '''
                # Weight Adjustment
                weight_col = self.weight_matrix[:, cur_neuron["id"]]
                for k in range(len(weight_col)):
                    if weight_col[k] != 0:

                        if self.fnc_learn_type == "BP":
                            self._tempWeightMatrix[k][j] = weight_col[k] - self.learnRate * cur_neuron["delta_value"] * cur_neuron["activation_value"]

                        elif self.fnc_learn_type == "ERS":
                            self._tempWeightMatrix[k][j] = weight_col[k] - self.learnRate * abs(1 - abs(weight_col[k])) * cur_neuron["delta_value"] * self.__sgn(cur_neuron["activation_value"])

                        elif self.fnc_learn_type == "ERS2":
                                self._tempWeightMatrix[k][j] = weight_col[k] - self.learnRate * abs(1 - abs(weight_col[k])) *  cur_neuron["delta_value"] * cur_neuron["activation_value"]
                '''

    def __fnc_propagate(self, index):
        weight_col = self.weight_matrix[:, index]

        netto_input = 0
        for i in range(len(weight_col)):
            netto_input += weight_col[i] * self.__get_neuron_value(i, "activation_value")

        return netto_input

    def __fnc_activate(self, index):


        if self.fnc_activate_type == "identity":
            return self.__fnc_propagate(index)

        elif self.fnc_activate_type == "log":
            return 1 / (1 + np.exp(-self.__fnc_propagate(index)))

        elif self.fnc_activate_type == "tanH":
            return (2 / (1 + np.exp(-2 * self.__fnc_propagate(index)))) - 1

    def __fnc_output(self, index):
        return self.__fnc_activate(index)

    def __set_neuron_value(self, neuron_id, value, value_type):


        if neuron_id < self.numInputNeurons:
            self.neurons[0][neuron_id][value_type] = value

        elif neuron_id >= self.numInputNeurons + self.numHiddenNeurons:
            self.neurons[len(self.neurons) - 1][neuron_id - self.numHiddenNeurons - self.numInputNeurons][value_type] = value

        else:
            remainder = neuron_id - self.numInputNeurons

            # Loop through all layers except for the first (input) and last (output) one
            for i in range(1, len(self.neurons) - 1):

                if remainder < len(self.neurons[i]):
                    self.neurons[i][remainder][value_type] = value
                    return
                else:
                    remainder -= len(self.neurons[i])

    def __get_neuron_value(self, neuron_id, value_type):

        if neuron_id < self.numInputNeurons:
            return self.neurons[0][neuron_id][value_type]

        elif neuron_id >= self.numInputNeurons + self.numHiddenNeurons:
            return self.neurons[len(self.neurons) - 1][neuron_id - self.numHiddenNeurons - self.numInputNeurons][value_type]

        else:
            remainder = neuron_id - self.numInputNeurons

            # Loop through all layers except for the first (input) and last (output) one
            for i in range(1, len(self.neurons) - 1):

                if remainder < len(self.neurons[i]):
                    return self.neurons[i][remainder][value_type]
                else:
                    remainder -= len(self.neurons[i])

    def __generate_weight_matrix(self, rnd_values_low, rnd_values_high):

        # Init Matrix with all zeros
        self.weight_matrix = np.zeros(shape=(self.numNeurons, self.numNeurons))

        # Create Connections between Input and Hidden Neurons and init them with random weights
        for i in range(self.numInputNeurons):
            for j in range(self.layers[1]):
                hidden_index = j + self.numInputNeurons
                rng = np.random.uniform(low=rnd_values_low, high=rnd_values_high, size=(1))
                self.weight_matrix[i][hidden_index] = rng[0]

        # Create Connections between the Hidden Neurons in different Layers
        for i in range(self.numHiddenLayers - 1):

            # Loop through neurons of current hidden layer
            for j in range(self.layers[i + 1]):

                # Loop through neurons in next hidden layer
                for k in range(self.layers[i + 2]):
                    rng = np.random.uniform(low=rnd_values_low, high=rnd_values_high, size=(1))
                    self.weight_matrix[j + self.numInputNeurons][k + self.numInputNeurons + self.layers[i + 1]] = rng[0]

        # Create Connections between Hidden and Output Neurons and init them with random weights
        for i in range(self.layers[len(self.layers) - 2]):
            hidden_index = i + self.numInputNeurons + self.numHiddenNeurons - self.layers[len(self.layers) - 2]
            for j in range(self.numOutputNeurons):
                out_idx = j + self.numInputNeurons + self.numHiddenNeurons
                rng = np.random.uniform(low=rnd_values_low, high=rnd_values_high, size=(1))
                self.weight_matrix[hidden_index][out_idx] = rng[0]

    @staticmethod
    def __calculate_error(target, output):
        """
        Calculate the Error of the Network
        """
        error = output - target
        return error

    @staticmethod
    def __sgn(value):
        if value < 0:
            return -1
        elif value == 0:
            return 0
        else:
            return 1

    def __derivative_activation(self, output):
        """
        derivative of the Activation Function
        """

        if self.fnc_activate_type == "identity":
            return 1

        elif self.fnc_activate_type == "log":
            return output * (1 - output)

        elif self.fnc_activate_type == "tanH":
            return 1 - (output * output)