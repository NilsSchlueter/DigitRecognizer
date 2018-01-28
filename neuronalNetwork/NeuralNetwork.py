import numpy as np
import random


class NeuralNetwork:
    def __init__(self, layers, weight_matrix=None, fnc_propagate_type="netto_input", fnc_activate_type="identity", fnc_learn_type="BP", fnc_output_type="identity", treshold=None,
            learn_rate=None, rnd_values_low=-1.0, rnd_values_high=1.0, number_epochs=1):

        # Misc attributes
        self.numLayers = len(layers)
        self.numNeurons = sum(layers)
        self.numInputNeurons = layers[0]
        self.numOutputNeurons = layers[self.numLayers - 1]
        self.numHiddenNeurons = layers[1]
        self.learnRate = learn_rate
        self.treshold = treshold
        self.epochs = number_epochs

        # Function type
        self.fnc_propagate_type = fnc_propagate_type
        self.fnc_activate_type = fnc_activate_type
        self.fnc_output_type = fnc_output_type
        self.fnc_learn_type = fnc_learn_type  # BP, ERS, ERS2

        # Set the functions to default values if no other values are provided
        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        # Init weights
        self.network_data = list()

        if weight_matrix is None:
            hidden_layer = [{'weights': [random.uniform(rnd_values_low, rnd_values_high) for i in range(self.numInputNeurons)]} for i in range(self.numHiddenNeurons)]
            self.network_data.append(hidden_layer)

            output_layer = [{'weights': [random.uniform(rnd_values_low, rnd_values_high) for i in range(self.numHiddenNeurons)]} for i in range(self.numOutputNeurons)]
            self.network_data.append(output_layer)
        else:

            neuron_data = []
            for layer in weight_matrix:
                neuron_data.append({"weights": layer})

            hidden_layer = neuron_data[:self.numHiddenNeurons]
            output_layer = neuron_data[self.numHiddenNeurons:]

            self.network_data.append(hidden_layer)
            self.network_data.append(output_layer)

    # -----------------------------------
    # PUBLIC FUNCTIONS
    # -----------------------------------
    def train(self, training_data):

        for epoch in range(self.epochs):
            sum_error = 0

            data_count = 0
            for row in training_data:

                data_count += 1

                # Read current input and output vectors
                input_vector = row[:self.numInputNeurons]
                output_vector = row[self.numInputNeurons:]

                outputs = self.__calculate_output(input_vector)

                sum_error += sum([(output_vector[i] - outputs[i]) ** 2 for i in range(len(output_vector))])

                self.__calc_errors(output_vector)
                self.__update_weights(row)

                if data_count % 100 == 0:
                    print("-- data: %d / %d" % (data_count, len(training_data)))

            print('finished epoch=%d / %d, learn rate=%.3f, error=%.3f' % (epoch + 1, self.epochs, self.learnRate, sum_error))

        # Training finished - save results
        self.__save_results()

    def test(self, test_data, print_results=False):

        f = None
        result_str = ""
        if print_results:
            f = open("test_results.txt", "w")
            f.write("===== Learn FNC: %s | Activation FNC: %s | Learn Rate: %s | Hidden Neurons: %s =====\n" % (self.fnc_learn_type, self.fnc_activate_type, self.learnRate, self.numHiddenNeurons))
            result_str += "===== Learn FNC: %s | Activation FNC: %s | Learn Rate: %s | Hidden Neurons: %s =====\n" % (
            self.fnc_learn_type, self.fnc_activate_type, self.learnRate, self.numHiddenNeurons)

        result = []
        correct_classifications = 0
        for row in test_data:

            row_result = []

            prediction = self.predict(row[:self.numInputNeurons])

            # Calculate predicted and target values
            predicted_value = self.__vector_to_digit(prediction)
            target_value = self.__vector_to_digit(row[self.numInputNeurons:])
            probs = self.__softmax(prediction)

            if predicted_value == target_value:
                correct_classifications += 1

            # append to results
            row_result.append(predicted_value)
            row_result.append(target_value)
            row_result.append(np.amax(probs))  # calculate Probability
            result.append(row_result)

            '''
            if f: 
                res_row = 'prediction: %s, actual: %s\n' % (prediction, row[self.numInputNeurons:])
                f.write(res_row)
                result_str += res_row
            '''

        if f:
            cor_class = 'Correct classifications: %d / %d\n\n' % (correct_classifications, len(test_data))
            f.write(cor_class)
            result_str += cor_class
            f.close()

        return result, result_str

    # -----------------------------------
    # LEARNING FUNCTIONS
    # -----------------------------------
    def __calculate_output(self, input_row):
        inputs = input_row
        for layer in self.network_data:
            new_inputs = []

            for neuron in layer:
                activation_value = self.__fnc_activate(neuron["weights"], inputs)
                neuron["output"] = self.__fnc_output(activation_value)
                new_inputs.append(neuron["output"])
            inputs = new_inputs
        return inputs

    def __calc_errors(self, target_values):

        # Go through each layer, starting with output layer
        for i in reversed(range(len(self.network_data))):
            layer = self.network_data[i]
            errors = list()

            # Error for neurons in HIDDEN layers
            if i != len(self.network_data) - 1:
                for j in range(len(layer)):
                    error = 0.0

                    # Go through neurons in previous layer and calc error
                    for neuron in self.network_data[i + 1]:
                        error += (neuron["weights"][j] * neuron["delta"])
                    errors.append(error)

            # Error for neurons in OUTPUT layer
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(target_values[j] - neuron["output"])

            # Calculate delta for each neuron in current layer, depending on learn function
            for j in range(len(layer)):
                neuron = layer[j]

                if self.fnc_learn_type == "BP":
                    neuron["delta"] = errors[j] * self.__derivative_activation(neuron["output"])

                else:
                    neuron["delta"] = errors[j]

    def __update_weights(self, row):

        for layer in range(len(self.network_data)):

            # If layer is input layer
            if layer == 0:
                prev_output = row[:self.numInputNeurons]

            # If layer is not input layer
            else:
                prev_output = [neuron["output"] for neuron in self.network_data[layer]]

            # Update weigth for each neuron, depending on the learn function
            for neuron in self.network_data[layer]:
                for j in range(len(prev_output)):

                    if self.fnc_learn_type == "BP":
                        neuron["weights"][j] += self.learnRate * neuron["delta"] * prev_output[j]

                    elif self.fnc_learn_type == "ERS":
                        neuron["weights"][j] += self.learnRate * abs(1 - abs(neuron["weights"][j])) * neuron["delta"] * self.__sgn(prev_output[j])

                    elif self.fnc_learn_type == "ERS2":
                        neuron["weights"][j] += self.learnRate * abs(1 - abs(neuron["weights"][j])) * neuron["delta"] * prev_output[j]

    def predict(self, input_row):
        outputs = self.__calculate_output(input_row)
        return outputs

    # -----------------------------------
    # 3 FUNCTIONS - PROPAGATE, ACTIVATE, OUTPUT
    # -----------------------------------
    def __fnc_activate(self, weights, inputs):

        if self.fnc_activate_type == "identity":
            return self.__fnc_propagate(weights, inputs)

        elif self.fnc_activate_type == "log":
            return 1 / (1 + np.exp(-self.__fnc_propagate(weights, inputs)))

        elif self.fnc_activate_type == "tanH":
            return (2 / (1 + np.exp(-2 * self.__fnc_propagate(weights, inputs)))) - 1

    def __derivative_activation(self, output):

        if self.fnc_activate_type == "identity":
            return 1

        elif self.fnc_activate_type == "log":
            return output * (1 - output)

        elif self.fnc_activate_type == "tanH":
            return 1 - (output * output)

    @staticmethod
    def __fnc_output(activation_value):
        return activation_value

    @staticmethod
    def __fnc_propagate(weights, inputs):
        netto_input = 0
        for i in range(len(weights)):
            netto_input += weights[i] * inputs[i]

        return netto_input

    # -----------------------------------
    # HELPER FUNCTIONS
    # -----------------------------------
    @staticmethod
    def __vector_to_digit(vector):
        result_digit = -1
        max_val = 0
        for i in range(len(vector)):
            if vector[i] > max_val:
                max_val = vector[i]
                result_digit = i
        return result_digit

    def __save_results(self):
        print("Saving results...")
        save_data = []

        f = open("weight_matrix_final.txt", "w")

        for layer in self.network_data:
            for neuron in layer:
                f.write(neuron["weights"].__str__())
                f.write("\n")
                save_data.append(neuron["weights"])

        f.close()

        # Write to np file
        np.save("weight_matrix_final", save_data)

    @staticmethod
    def __sgn(value):
        if value < 0:
            return -1
        elif value == 0:
            return 0
        else:
            return 1

    @staticmethod
    def __softmax(data):
        return np.exp(data) / np.sum(np.exp(data))

