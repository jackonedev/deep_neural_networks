import numpy as np

class NeuralNetwork:
    def __init__(self, layers_size, activation_functions):
        """
        Initializes the neural network.

        Args:
            layers_size (list): A list of integers representing the number of neurons in each layer.
            activation_functions (list): A list of strings representing the activation function for each layer (excluding the input layer).
        """
        self.num_layers = len(layers_size)
        self.layers_size = layers_size
        self.activation_functions = activation_functions
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

    def _initialize_weights(self):
         """
         Initializes weights with random values.
         Using He initialization for better convergence.

         Returns:
             list: A list of weight matrices for each layer.
         """
         weights = []
         for i in range(self.num_layers - 1):
             #He initialization
             limit = np.sqrt(2 / self.layers_size[i])
             w = np.random.normal(0, limit, (self.layers_size[i], self.layers_size[i+1]))
             weights.append(w)
         return weights

    def _initialize_biases(self):
        """
        Initializes biases with zeros.

        Returns:
            list: A list of bias vectors for each layer.
        """
        biases = []
        for i in range(self.num_layers - 1):
            b = np.zeros((1, self.layers_size[i+1]))
            biases.append(b)
        return biases

    def _sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def _relu(self, x):
        """
        ReLU activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying the ReLU function.
        """
        return np.maximum(0, x)

    def _softmax(self, x):
        """
        Softmax activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array after applying the softmax function.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _activate(self, x, activation):
         """
         Applies the specified activation function.

         Args:
             x (numpy.ndarray): Input array.
             activation (str): The name of the activation function ('sigmoid', 'relu', or 'softmax').

         Returns:
             numpy.ndarray: Output array after applying the activation function.
         """
         if activation == 'sigmoid':
             return self._sigmoid(x)
         elif activation == 'relu':
             return self._relu(x)
         elif activation == 'softmax':
              return self._softmax(x)
         else:
             raise ValueError("Invalid activation function")

    def forward_propagation(self, input_data):
        """
        Performs forward propagation through the network.

        Args:
            input_data (numpy.ndarray): Input data array.

        Returns:
            numpy.ndarray: Output of the network.
        """
        a = input_data
        for i in range(self.num_layers - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self._activate(z, self.activation_functions[i])
        return a

    def predict(self, input_data):
        """
        Makes predictions using the trained network.

        Args:
            input_data (numpy.ndarray): Input data array.

        Returns:
             numpy.ndarray: Predicted output.
        """
        return self.forward_propagation(input_data)

if __name__ == '__main__':
    # Example usage:
    layers_size = [2, 3, 2] # Input layer (2 neurons), hidden layer (3 neurons), output layer (2 neurons)
    activation_functions = ['relu', 'softmax']  # ReLU for hidden layer, softmax for output layer
    input_data = np.array([[0.5, 0.2], [0.8, 0.9]])

    # Create and use the neural network
    nn = NeuralNetwork(layers_size, activation_functions)
    output = nn.predict(input_data)
    print("Input data:")
    print(input_data)
    print("Predicted output:")
    print(output)
