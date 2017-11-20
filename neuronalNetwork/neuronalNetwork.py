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
        self.numHiddenNeurons = layers[1]
        self.learnRate = learnRate;

        # Set the functions to default values if no other values are provided
        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        # Use defined weight matrix if given else init random weight matrix
        self._weightMatrix = np.random.uniform(low=-1, high=1, size=(self.numNeurons, self.numNeurons)) if weight_matrix is None else weight_matrix
        for i in range(self.numNeurons):
            for j in range(self.numNeurons):
                if(self.weight_matrix[i][j] !=0):
                    randNum = np.random.uniform(low=-1, high=1, size=(1));
                    self.weight_matrix[i][j] = randNum[0]

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
        new_weightMatrix = np.zeros(shape=(self.numNeurons,self.numNeurons))

        for i in range(len(trainingData)):
            random =np.random.randint(len(trainingData), size=(1))
            i=random[0]
            sumerror =0
            inputVector = trainingData[i]["input"]
            outputVector = trainingData[i]["output"]

            # Set input pattern to neurons in first layer
            for j in range(len(inputVector)):
                self.neurons[j] = inputVector[j]

            # Calculate output for the other neurons
            for k in range(len(inputVector), self.numNeurons):
                self.neurons[k] = self.__fnc_output(k)
                
            #print("%s | %s" % (inputVector, self.neurons[-self.numOutputNeurons:]))
           
            #Backpropagation for Output Layer
            for l in range(self.numOutputNeurons):
                activationIndex = self.numNeurons-l-1
                #Delta Calculation
                error = 0;
                error= self.__calculateError(outputVector[self.numOutputNeurons-l-1],self.neurons[activationIndex])
                sumerror +=error;
                derivativeValue = self.__derivativeActivation(self.neurons[activationIndex])
                self.delta[activationIndex] = error * derivativeValue
                #print("Error : %s"%(error))
                
                #Weight Adjustment
                weightCol = self._weightMatrix[:, activationIndex]
                for i in range(len(weightCol)):
                    if(weightCol[i]!=0):
                        #print("%s - %s * %s * %s"%(weightCol[i] , self.learnRate ,self.delta[activationIndex] ,self.neurons[i]))
                        new_weightMatrix[i][activationIndex]= weightCol[i] - self.learnRate * self.delta[activationIndex] * self.neurons[i]
                       
                        
            #print(self.delta)
            #print(self.weight_matrix)
            #print(new_weightMatrix)
                        
            #Backpropagation for Hidden Layer
            for m in range(self.numHiddenNeurons):
                activationIndex = self.numInputNeurons+m
                #print(activationIndex)
                #Sum Error of Layer before
                error = 0;
                weightRow = self._weightMatrix[ activationIndex,:]
                for n in range(len(weightRow)):
                    if(weightRow[n]!=0):
                        error += self.delta[n]*weightRow[n]
                        #print(weightRow);
                        
                #print("BackpropagationError : %s"%(error))
                derivativeValue = self.__derivativeActivation(self.neurons[activationIndex])
                self.delta[activationIndex] = error * derivativeValue
                        
                #Weight Adjustment
                weightCol = self._weightMatrix[:, activationIndex]
                
                for o in range(len(weightCol)):
                    if(weightCol[o]!=0):
                        #print("%s - %s * %s * %s"%(weightCol[i] , self.learnRate ,self.delta[activationIndex] ,self.neurons[i]))
                        new_weightMatrix[o][activationIndex]= weightCol[o] - self.learnRate * self.delta[activationIndex] * self.neurons[o]
                        #print("Ergebnis : %s" %(weightCol[i]))
                        
                #print(new_weightMatrix)
            self.weight_matrix = new_weightMatrix; 
            return sumerror;
                        
                
                
                        
            
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
            
        #print("Netto Input : %s" % (netto_input));
        return netto_input

    def __fnc_activate(self, index):
        """
        Activation function
        """
        neuronAktivierung = (1/(1+np.exp(-self.__fnc_propagate(index))));
        #print("Neuron Aktivierung : %s"%(neuronAktivierung));
        return neuronAktivierung;
    def __fnc_output(self, index):
        """
        Output function.
        """
        return self.__fnc_activate(index)


