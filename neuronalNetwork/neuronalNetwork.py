import numpy as np
from random import randint

class NeuronalNetwork:

    def __init__(self, layers, weight_matrix=None, fnc_propagate_type=None, fnc_activate_type=None, fnc_output_type=None, learnRate=0.5):

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
        self.learnRate = learnRate

        # Function type
        self.fnc_propagate_type = fnc_propagate_type
        self.fnc_activate_type = fnc_activate_type
        self.fnc_output_type = fnc_output_type

        # Set the functions to default values if no other values are provided
        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        # Use defined weight matrix if given else init random weight matrix
        self._weightMatrix = np.random.uniform(low=-1, high=1, size=(self.numNeurons, self.numNeurons)) if weight_matrix is None else weight_matrix
        for i in range(self.numNeurons):
            for j in range(self.numNeurons):
                if(self.weight_matrix[i][j] !=0):
                    randNum = np.random.uniform(low=-1, high=1, size=(1))
                    self.weight_matrix[i][j] = randNum[0]

        self._tempWeightMatrix = weight_matrix
        self.inputValues = []
        self.targetValues = []

        # Array which has the current output of each neuron
        self.neurons = np.zeros(self.numNeurons)
        self.delta = np.zeros(self.numNeurons)


    @property
    def weight_matrix(self):
        return self._weightMatrix

    @weight_matrix.setter
    def weight_matrix(self, value):
        self._weightMatrix = value

    def train(self, trainingData, maxIterations):
        """
        Trains the network with the given training data.
        """
        new_weightMatrix = np.zeros(shape=(self.numNeurons,self.numNeurons))

        iteration = 0
        while iteration < maxIterations:

            iteration += 1
            sumerror = 0

            # Get random training data
            i = randint(0, len(trainingData) - 1)

            # Read current input and output vectors
            inputVector = trainingData[i]["input"]
            outputVector = trainingData[i]["output"]

            # Set input pattern to neurons in first layer
            for j in range(len(inputVector)):
                self.neurons[j] = inputVector[j]

            # Calculate output for the other neurons
            for k in range(len(inputVector), self.numNeurons):
                self.neurons[k] = self.__fnc_output(k)

            # Change weight matrix
            self.__backpropagation(outputVector)
            self.weight_matrix = self._tempWeightMatrix

    def __backpropagation(self, output_vector):
        self.__backpropagationOutput(output_vector)
        self.__backpropagationHidden()

    def __backpropagationOutput(self, outputVector):

        # Backpropagation for Output Layer
        for l in range(self.numOutputNeurons):

            # Index of current output neuron
            activation_index = self.numNeurons - l - 1

            # Get target and actual value
            target_value = outputVector[self.numOutputNeurons - l - 1]
            actual_value = self.neurons[activation_index]

            # Delta Calculation
            error = self.__calculateError(target_value, actual_value)

            derivative_value = self.__derivativeActivation(actual_value)
            self.delta[activation_index] = error * derivative_value

            # Weight Adjustment
            weight_col = self._weightMatrix[:, activation_index]
            for i in range(len(weight_col)):
                if weight_col[i] != 0:
                    self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * self.delta[activation_index] * self.neurons[i]

    def __backpropagationHidden(self):

        for l in range(self.numHiddenNeurons):

            # Index of current output neuron
            activation_index = self.numInputNeurons + l
            actual_value = self.neurons[activation_index]
            error = 0

            # Delta Calculation
            weight_row = self._weightMatrix[activation_index, :]
            for n in range(len(weight_row)):
                if weight_row[n] != 0:
                    error += self.delta[n] * weight_row[n]

            derivative_value = self.__derivativeActivation(actual_value)
            self.delta[activation_index] = error * derivative_value

            # Weight Adjustment
            weight_col = self._weightMatrix[:, activation_index]
            for i in range(len(weight_col)):
                if weight_col[i] != 0:
                    self._tempWeightMatrix[i][activation_index] = weight_col[i] - self.learnRate * self.delta[activation_index] * self.neurons[i]

    def __calculateError(self, target, output):
        """
        Calculate the Error of the Network
        """
        error = output-target
        return error

    def __derivativeActivation(self, output):
        """
        derivative of the Activation Function
        """

        if self.fnc_activate_type == "identity":
            return 1

        elif self.fnc_activate_type == "log":
            return output * (1 - output)

        elif self.fnc_activate_type == "tanH":
            return 1 - (output * output)

    def test(self, testData):
        """
        Tests the network with the given test data.
        """
        for i in range(len(testData)):

            sumerror =0
            inputVector = testData[i]["input"]
            outputVector = testData[i]["output"]

            # Set input pattern to neurons in first layer
            for j in range(len(inputVector)):
                self.neurons[j] = inputVector[j]

            # Calculate output for the other neurons
            for k in range(len(inputVector), self.numNeurons):
                self.neurons[k] = self.__fnc_output(k)
            for l in range(self.numOutputNeurons):
                outnumber = self.numNeurons - l -1;
                print("Output : %s Target : %s"%(self.neurons[outnumber],outputVector[self.numOutputNeurons-l-1]));

        pass

    def __fnc_propagate(self, index):
        """
        Propagation function
        """

        weightCol = self._weightMatrix[:, index]

        netto_input = 0
        for i in range(len(weightCol)):
            netto_input += weightCol[i] * self.neurons[i]

        #print("Netto Input : %s" % (netto_input));
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


