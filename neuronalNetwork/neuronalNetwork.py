import numpy as np


class NeuronalNetwork:

    def __init__(self, layers, weight_matrix=None, fnc_propagate_type=None, fnc_activate_type=None, fnc_output_type=None):

        """
        Inits a new Neuronal Network.

        :param layers: Array containing the number of hidden layers and the number of neurons in each layer
        """

        self.numLayers = len(layers)
        self.numNeurons = sum(layers)

        self.fnc_propagate_type = "netto_input" if fnc_propagate_type is None else fnc_propagate_type
        self.fnc_activate_type = "identity" if fnc_activate_type is None else fnc_activate_type
        self.fnc_output_type = "identity" if fnc_output_type is None else fnc_output_type

        # Use defined weight matrix if given else init random weight matrix
        self._weightMatrix = np.random.rand(self.numNeurons, self.numNeurons) if weight_matrix is None else weight_matrix

        self.inputNeurons = []
        self.outputNeurons = []

    @property
    def weight_matrix(self):
        return self._weightMatrix

    @weight_matrix.setter
    def weight_matrix(self, value):
        self._weightMatrix = value



    def train(self):
        """
        Trains the network with the given training data.
        """

        pass

    def test(self):
        """
        Tests the network with the given test data.
        """
        pass


    def __backpropagation(self):

        """
        Backpropagation method.
        """

        pass

    def __fnc_propagate(self):

        """
        Propagation function
        """

        pass

    def __fnc_activate(self):

        """
        Activation function
        """

        pass

    def __fnc_output(self):

        """
        Output function.
        """

        pass


