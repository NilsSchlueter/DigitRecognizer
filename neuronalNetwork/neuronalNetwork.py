import numpy as np
import sys
from random import randint

class NeuronalNetwork:

    def __init__(self, layers, weight_matrix=None, fnc_propagate_type=None, fnc_activate_type=None, fnc_learn_type="BP",
                 fnc_output_type=None, treshold=None, learn_rate=0.5, rnd_values_low=-1.0, rnd_values_high=1.0):

        """
        Inits a new Neuronal Network.

        :param layers: Array containing the number of hidden layers and the number of neurons in each layer
        """

        # Misc attributes
        self.numLayers = len(layers)
        self.numNeurons = sum(layers)
        self.numInputNeurons = layers[0]
        self.numOutputNeurons = layers[self.numLayers-1]
        self.numHiddenNeurons = layers[1]
        self.learnRate = learn_rate
        self.treshold = treshold

        # Function type
        self.fnc_propagate_type = fnc_propagate_type
        self.fnc_activate_type = fnc_activate_type
        self.fnc_output_type = fnc_output_type
        self.fnc_learn_type = fnc_learn_type #BP, ERS, ERS2

        # Set the functions to default values if no other values are provided
        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        # RandomMatrix FeedForward
        if weight_matrix is None:
            self.weight_matrix = np.zeros(shape=(self.numNeurons, self.numNeurons))
            for i in range(self.numInputNeurons):
                for j in range(self.numHiddenNeurons):
                    hiddenIdx = j + self.numInputNeurons
                    rng = np.random.uniform(low=rnd_values_low, high=rnd_values_high, size=(1))
                    self.weight_matrix[i][hiddenIdx] = rng[0]

            for i in range(self.numHiddenNeurons):
                hiddenIdx = i + self.numInputNeurons
                for j in range(self.numOutputNeurons):
                    outIdx = j + self.numInputNeurons + self.numHiddenNeurons
                    rng = np.random.uniform(low=rnd_values_low, high=rnd_values_high, size=(1))
                    self.weight_matrix[hiddenIdx][outIdx] = rng[0]
        else:
            self.weight_matrix = weight_matrix


        # If self.weight_matrix changes, self._tempWeightMatrix also changes, but not the other way around!
        # Thus we don't need to update self._tempWeightMatrix after each step, as we already update self.weight_matrix
        self._tempWeightMatrix = self.weight_matrix
        self.inputValues = []
        self.targetValues = []

        # Array which has the current output of each neuron
        self.neurons = np.zeros(self.numNeurons)
        self.delta = np.zeros(self.numNeurons)
        self.output = np.zeros(self.numOutputNeurons)

    def train(self, training_data, max_iterations):
        """
        Trains the network with the given training data.
        """

        try:
            # Redirect print to file
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
                    self.neurons[j] = input_vector[j]

                increase_error = False

                # Calculate output for the other neurons
                for k in range(len(input_vector), self.numNeurons):
                    # Calculate Activation
                    self.neurons[k] = self.__fnc_output(k)

                    # Calculate output neurons with treshold
                    if k >= (self.numHiddenNeurons + self.numInputNeurons):
                        index = k - self.numHiddenNeurons - self.numInputNeurons
                        if self.treshold is not None:
                            self.output[index] = 1 if self.neurons[k] > self.treshold else 0
                            if self.output[index] != output_vector[index]:
                                increase_error = True
                        else:
                            self.output[index] =self.neurons[k]





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
                    f.write(str(error_sum/100))
                    print("Iteration %s/%s" % (iteration, max_iterations))
                    print("Wrong classifications in the last 100 iterations: " + str(error_sum/100))
                    f.write("\n\n")

                    error_sum = 0

            f.close()

            # USE WITH CAUTION
            # np.set_printoptions(threshold=np.nan)
            f_final = open("weight_matrix_final.txt", "w")
            f_final.write(str(self.weight_matrix))
            f_final.close()

            # Save as numpy
            np.save("weight_matrix_final_np", self.weight_matrix)

        except KeyboardInterrupt:
            # np.set_printoptions(threshold=np.nan)
            f_final = open("weight_matrix_final.txt", "w")
            f_final.write(str(self.weight_matrix))
            f_final.close()

            np.save("weight_matrix_final_np", self.weight_matrix)
            sys.exit()

    def test(self, test_data):
        """
        Tests the network with the given test data.
        """

        resultStr = ""

        count = 0
        for i in range(len(test_data)):

            increase_count = True

            input_vector = test_data[i]["input"]
            output_vector = test_data[i]["output"]

            # Set input pattern to neurons in first layer
            for j in range(len(input_vector)):
                self.neurons[j] = input_vector[j]

            # Calculate output for the other neurons
            for k in range(len(input_vector), self.numNeurons):
                self.neurons[k] = self.__fnc_output(k)

                # Calculate output neurons with treshold
                if k >= (self.numHiddenNeurons + self.numInputNeurons):
                    if self.treshold is not None:
                        self.output[k - self.numHiddenNeurons - self.numInputNeurons] = 1 if self.neurons[k] > self.treshold else 0
                    else:
                        self.output[k - self.numHiddenNeurons - self.numInputNeurons] = self.neurons[k]


            out = np.zeros(self.numOutputNeurons)
            resultDigit = -1
            networkDigit = 0
            max_val = 0
            for l in range(self.numOutputNeurons):
                out[l] = self.neurons[l + self.numInputNeurons + self.numHiddenNeurons]
                if self.output[l] != output_vector[l] and self.treshold is not None:
                    increase_count = False
                if self.treshold is None:
                    if self.output[l] > max_val:
                        max_val = self.output[l]
                        networkDigit = l
                    if output_vector[l] == 1:
                        resultDigit = l



            if networkDigit == resultDigit:
                count += 1

            resultStr += "Outputvector: %s Targetvector: %s Resultvector: %s\n" % (out, output_vector, self.output)

        percent = count/len(test_data)
        resultStr += "%s wurden erfolgreich erkannt" % (percent)

        f_final = open("test_results.txt", "w")
        f_final.write(resultStr)
        f_final.close()

        return resultStr

    def test_single_digit(self, digit_data):

        input_vector = digit_data["input"]

        # Set input pattern to neurons in first layer
        for j in range(len(input_vector)):
            self.neurons[j] = input_vector[j]

        # Calculate output for the other neurons
        for k in range(len(input_vector), self.numNeurons):
            self.neurons[k] = self.__fnc_output(k)

            # Calculate output neurons with treshold
            if k >= (self.numHiddenNeurons + self.numInputNeurons):
                if self.treshold is not None:
                    self.output[k - self.numHiddenNeurons - self.numInputNeurons] = 1 if self.neurons[
                                                                                             k] > self.treshold else 0
                else:
                    self.output[k - self.numHiddenNeurons - self.numInputNeurons] = self.neurons[k]
        return self.output

    def __fnc_learn(self, output_vector):
        self.__fnc_learn_output(output_vector)
        self.__fnc_learn_hidden()

    def __fnc_learn_output(self, output_vector):

        # Backpropagation for Output Layer
        sumerror = 0
        for l in range(self.numOutputNeurons):

            # Index of current output neuron
            activation_index = self.numNeurons - l - 1

            # Get target and actual value
            target_value = output_vector[self.numOutputNeurons - l - 1]
            actual_value = self.neurons[activation_index]
            real_output = self.output[self.numOutputNeurons - l - 1]

            # Calculate Error
            error = self.__calculate_error(target_value, real_output)
            sumerror += error

            # Delta Calculation depending on learn function
            if self.fnc_learn_type == "BP":
                derivative_value = self.__derivative_activation(actual_value)
                self.delta[activation_index] = error * derivative_value
            elif self.fnc_learn_type == "ERS" or self.fnc_learn_type == "ERS2":
                self.delta[activation_index] = error

            # Weight Adjustment
            weight_col = self.weight_matrix[:, activation_index]

            for i in range(len(weight_col)):
                if weight_col[i] != 0:

                    if self.fnc_learn_type == "BP":
                        self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * self.delta[activation_index] * self.neurons[i]
                    elif self.fnc_learn_type == "ERS":
                        self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * abs(1 - abs(weight_col[i])) * self.delta[activation_index] * self.__sgn(self.neurons[i])
                    elif self.fnc_learn_type == "ERS2":
                        self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * abs(1 - abs(weight_col[i])) * self.delta[activation_index] * self.neurons[i]

    def __fnc_learn_hidden(self):

        for l in range(self.numHiddenNeurons):

            # Index of current output neuron
            activation_index = self.numInputNeurons + l
            actual_value = self.neurons[activation_index]
            error = 0

            # Delta Calculation
            weight_row = self.weight_matrix[activation_index, :]
            for n in range(len(weight_row)):
                if weight_row[n] != 0:
                    error += self.delta[n] * weight_row[n]

            if self.fnc_learn_type == "BP":
                derivative_value = self.__derivative_activation(actual_value)
                self.delta[activation_index] = error * derivative_value
            elif self.fnc_learn_type == "ERS" or self.fnc_learn_type == "ERS2":
                self.delta[activation_index] = error

            # Weight Adjustment
            weight_col = self.weight_matrix[:, activation_index]
            for i in range(len(weight_col)):
                if weight_col[i] != 0:

                    if self.fnc_learn_type == "BP":
                        self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * self.delta[activation_index] * self.neurons[i]
                    elif self.fnc_learn_type == "ERS":
                        self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * abs(1 - abs(weight_col[i])) * self.delta[activation_index] * self.__sgn(self.neurons[i])
                    elif self.fnc_learn_type == "ERS2":
                        self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * abs(1 - abs(weight_col[i])) * self.delta[activation_index] * self.neurons[i]

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

    def __fnc_propagate(self, index):
        """
        Propagation function
        """
        weight_col = self.weight_matrix[:, index]

        netto_input = 0
        for i in range(len(weight_col)):
            netto_input += weight_col[i] * self.neurons[i]

        return netto_input

    def __fnc_activate(self, index):
        """
        Activation function
        """

        if self.fnc_activate_type == "identity":
            return self.__fnc_propagate(index)

        elif self.fnc_activate_type == "log":
            return 1 / (1 + np.exp(-self.__fnc_propagate(index)))

        elif self.fnc_activate_type == "tanH":
            return (2 / (1 + np.exp(-2 * self.__fnc_propagate(index)))) - 1

    def __fnc_output(self, index):
        """
        Output function.
        """
        return self.__fnc_activate(index)
