import numpy as np


class NeuronalNetwork:

    def __init__(self, layers, weight_matrix=None, fnc_propagate_type=None, fnc_activate_type=None, fnc_output_type=None,learnRate=0.5):

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
        self.learnRate = learnRate;

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
        self.delta = np.zeros(self.numNeurons)


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
           
           #Backpropagation for Output Layer
            for l in range(self.numOutputNeurons):
                activationIndex = self.numNeurons-l-1
                #Delta Calculation
                error= self.__calculateError(outputVector[self.numOutputNeurons-l-1],self.neurons[activationIndex])
                derivativeValue = self.__derivativeActivation(self.neurons[activationIndex])
                print ("Output:%s Delta:  %s*%s"%(self.neurons[activationIndex],error,derivativeValue))
                self.delta[activationIndex] = error * derivativeValue
                
                #Weight Adjustment
                weightCol = self._weightMatrix[:, activationIndex]
                for i in range(len(weightCol)):
                    if(weightCol[i]!=0):
                        print("%s - %s * %s * %s"%(weightCol[i] , self.learnRate ,self.delta[activationIndex] ,self.neurons[i]))
                        weightCol[i]= weightCol[i] - self.learnRate * self.delta[activationIndex] * self.neurons[i]
                        print("Ergebnis : %s" %(weightCol[i]))
                        
            
    def __calculateError(self,target,output):
        """
        Calculate the Error of the Network
        """
        error = output-target 
        return error     
    
    
    def __derivativeActivation(self,output):
        """
        derivative of the Activation Function
        """
        return output*(1-output)        

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


