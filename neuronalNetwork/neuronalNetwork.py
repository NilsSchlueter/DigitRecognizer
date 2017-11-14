import numpy as np


class NeuronalNetwork:

    def __init__(self, layers, weight_matrix=None, fnc_propagate_type=None, fnc_activate_type=None, fnc_output_type=None):

        """
        Inits a new Neuronal Network.

        :param layers: Array containing the number of hidden layers and the number of neurons in each layer
        """

        # Misc attributes
        self.numLayers = len(layers)
        self.numNeurons = sum(layers)
        self.numInputNeurons = layers[0];
        self.numOutputNeurons = layers[self.numLayers-1];
        self.numHiddenNeurons = self.numOutputNeurons - self.numInputNeurons

        # Set the functions to default values if no other values are provided
        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        # Use defined weight matrix if given else init random weight matrix
        self._weightMatrix = np.random.uniform(low=-1, high=1, size=(self.numNeurons, self.numNeurons)) if weight_matrix is None else weight_matrix

        self.inputValues = []
        self.targetValues = []

        # Array which has the current output of each neuron
        self.neurons = np.zeros(self.numNeurons)


    @property
    def weight_matrix(self):
        return self._weightMatrix

    @weight_matrix.setter
    def weight_matrix(self, value):
        self._weightMatrix = value

    def train(self, trainingData):
        """
        Trains the network with the given training data.
        """

        for i in range(len(trainingData)):

            inputVector = trainingData[i]["input"]
            outputVector = trainingData[i]["output"]

            # Set input pattern to neurons in first layer
            for j in range(len(inputVector)):
                self.neurons[j] = inputVector[j]

            # Calculate output for the other neurons
            for k in range(len(inputVector), self.numNeurons):
                self.neurons[k] = self.__fnc_output(k)
                
            print("%s | %s" % (inputVector, self.neurons[-self.numOutputNeurons:]))
            error = 0
            for l in range(self.numOutputNeurons):
                error += self.__calculateError(outputVector[l],self.neurons[self.numHiddenNeurons+l])
            print("Error : %s" %(error))
            
    def __calculateError(self,target,output):
        """
        Calculate the Error of the Network
        """
        error = 1/2*(pow(target-output,2)) 
        return error     
    
        pass

    def test(self, testData):
        """
        Tests the network with the given test data.
        """
        pass

    def __backpropagation(self):
        """
        Backpropagation method.
        """

        pass

    def __fnc_propagate(self, index):
        """
        Propagation function
        """

        weightCol = self._weightMatrix[:, index]

        netto_input = 0
        for i in range(len(weightCol)):
            netto_input += weightCol[i] * self.neurons[i]
            
        print("Netto Input : %s" % (netto_input));
        return netto_input

    def __fnc_activate(self, index):
        """
        Activation function
        """
        neuronAktivierung = (1/(1+np.exp(-self.__fnc_propagate(index))));
        print("Neuron Aktivierung : %s"%(neuronAktivierung));
        return neuronAktivierung;
    def __fnc_output(self, index):
        """
        Output function.
        """
        return self.__fnc_activate(index)


