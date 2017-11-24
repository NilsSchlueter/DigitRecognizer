import numpy as np
from random import randint


class NeuronalNetwork:

    def __init__(self, layers, weight_matrix=None, fnc_propagate_type=None, fnc_activate_type=None,
                 fnc_output_type=None, learn_rate=0.5):

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

        # Function type
        self.fnc_propagate_type = fnc_propagate_type
        self.fnc_activate_type = fnc_activate_type
        self.fnc_output_type = fnc_output_type

        # Set the functions to default values if no other values are provided
        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        # RandomMatrix with everything
        #self.weight_matrix = np.random.uniform(low=-0.5, high=0.5, size=(self.numNeurons, self.numNeurons)) if weight_matrix is None else weight_matrix
        
        # RandomMatrix FeedForward
        self.weight_matrix = np.zeros(shape=(self.numNeurons,self.numNeurons))
        for i in range(self.numInputNeurons):
            for j in range(self.numHiddenNeurons):
                hiddenIdx = j + self.numInputNeurons
                rng = np.random.uniform(low=-0.5, high=0.5, size=(1)) 
                self.weight_matrix[i][hiddenIdx] = rng[0]
        for i in range(self.numHiddenNeurons):
            hiddenIdx = i + self.numInputNeurons
            for j in range(self.numOutputNeurons):
                outIdx = j + self.numInputNeurons + self.numHiddenNeurons
                rng = np.random.uniform(low=-0.5, high=0.5, size=(1)) 
                self.weight_matrix[hiddenIdx][outIdx] = rng[0]
                
        print(self.weight_matrix)
            
            

        # If self.weight_matrix changes, self._tempWeightMatrix also changes, but not the other way around!
        # Thus we don't need to update self._tempWeightMatrix after each step, as we already update self.weight_matrix
        self._tempWeightMatrix = self.weight_matrix
        self.inputValues = []
        self.targetValues = []

        # Array which has the current output of each neuron
        self.neurons = np.zeros(self.numNeurons)
        self.delta = np.zeros(self.numNeurons)
        
  
       
        
        

    def train(self, training_data, max_iterations):
        """
        Trains the network with the given training data.
        """

        iteration = 0
        while iteration < max_iterations:

            iteration += 1
            print("Iteration %s/%s"%(iteration,max_iterations))

            # Get random training data
            i = randint(0, len(training_data) - 1)

            # Read current input and output vectors
            input_vector = training_data[i]["input"]
            output_vector = training_data[i]["output"]
            

            # Set input pattern to neurons in first layer
            for j in range(len(input_vector)):
                self.neurons[j] = input_vector[j]

            # Calculate output for the other neurons
            for k in range(len(input_vector), self.numNeurons):
                self.neurons[k] = self.__fnc_output(k)

            # Change weight matrix
            self.__backpropagation(output_vector)
            self.weight_matrix = self._tempWeightMatrix

    def __backpropagation(self, output_vector):
        self.__backpropagation_output(output_vector)
        self.__backpropagation_hidden()

    def __backpropagation_output(self, output_vector):

        # Backpropagation for Output Layer
        for l in range(self.numOutputNeurons):

            # Index of current output neuron
            activation_index = self.numNeurons - l - 1

            # Get target and actual value
            target_value = output_vector[self.numOutputNeurons - l - 1]
            actual_value = self.neurons[activation_index]

            # Delta Calculation
            error = self.__calculate_error(target_value, actual_value)

            derivative_value = self.__derivative_activation(actual_value)
            self.delta[activation_index] = error * derivative_value

            # Weight Adjustment
            weight_col = self.weight_matrix[:, activation_index]
            for i in range(len(weight_col)):
                if weight_col[i] != 0:
                    self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * self.delta[activation_index] * self.neurons[i]

    def __backpropagation_hidden(self):

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

            derivative_value = self.__derivative_activation(actual_value)
            self.delta[activation_index] = error * derivative_value

            # Weight Adjustment
            weight_col = self.weight_matrix[:, activation_index]
            for i in range(len(weight_col)):
                if weight_col[i] != 0:
                    self._tempWeightMatrix[i][activation_index] \
                        = weight_col[i] - self.learnRate * self.delta[activation_index] * self.neurons[i]

    @staticmethod
    def __calculate_error(target, output):
        """
        Calculate the Error of the Network
        """
        error = output - target
        return error

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

    def test(self, test_data):
        """
        Tests the network with the given test data.
        """
        count = 0
        for i in range(len(test_data)):

            input_vector = test_data[i]["input"]
            output_vector = test_data[i]["output"]

            # Set input pattern to neurons in first layer
            for j in range(len(input_vector)):
                self.neurons[j] = input_vector[j]

            # Calculate output for the other neurons
            for k in range(len(input_vector), self.numNeurons):
                self.neurons[k] = self.__fnc_output(k)

            out = np.zeros(self.numOutputNeurons)
            idxO = 0
            tempO= 0
            for l in range(self.numOutputNeurons):
                output = self.numInputNeurons+self.numHiddenNeurons +l
                #out[l] = 1 if self.neurons[output]  > 0.5 else 0 
                out[l] = self.neurons[output] 
                if(out[l]>tempO):
                    tempO = out[l]
                    idxO = l
                if(output_vector[l]==1):
                    idxT = l
            
            
            print("Target : %s Output : %s" % (idxT,idxO))
            if(idxT == idxO):
                count +=1
            #print("Output : %s Target : %s" % (out, output_vector))

        percent = count/len(test_data)
        print("%s wurden erfolgreich erkannt"%(percent))

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
            return 2 / (1 + np.exp(-2 * self.__fnc_propagate(index))) - 1

    def __fnc_output(self, index):
        """
        Output function.
        """
        return self.__fnc_activate(index)
