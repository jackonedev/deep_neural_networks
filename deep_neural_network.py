import pickle, os, copy
import numpy as np
import matplotlib.pyplot as plt
import h5py
from originals.dnn_app_utils_v3 import sigmoid, sigmoid_backward, relu, relu_backward

outline = """   PASOS A SEGUIR PARA CONSTRUIR UNA RED NEURONAL PROFUNDA

1) Inicializar parámetros / Definir hiperparámetros
    1.1) Inicialización una red neuronal de 2 capas
    1.2) Inicialización una red neuronal L-capa
2) Implementación de forward propagation
    2.1) Linear Forward
    2.2) Linear-Activation Forward (ReLU, sigmoid)
    2.3) Combinación de 2.1 y 2.2
    2.3) Stack de L-1 capas de 2.3 y añadir la activación sigmoid en la última capa
3) Función de costo
4) Implementación de backward propagation
    4.1) Completa la capa LINEAR hacia atrás
    4.2) El gradiente de la función de activación ReLU/sigmoid
    4.3) Combinación de 4.1 y 4.2
    4.3) Stack de L-1 capas de 4.3 y añadir la activación sigmoid en la última capa
5) Actualización de parámetros
"""


###  1.1 Inicializacion red de dos capas (1 hidden, 1 output) ###
def initialize_parameters(n_x: int, n_h: int, n_y: int) -> dict:
    """

    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))    
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters 

###  1.2 Inicializacion red de L capas ###
def initialize_parameters_deep(layer_dims: list) -> dict:
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        # parameters['W' + str(l)] = ...
        # parameters['b' + str(l)] = ...
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

### 2.1 Implementación de linear forward ###
def linear_forward(A: np.array, W: np.array, b: np.array) -> tuple:
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    # TODO: evita el broadcasting: 
    # assert W.shape[1] == A.shape[0], "The number of columns in the first matrix must be equal to the number of rows in the second matrix for dot product."
    
    Z = np.dot(W, A) + b
   
    cache = (A, W, b)
    
    return Z, cache

### 2.2 Implementación de linear-activation forward ###
def linear_activation_forward(A_prev: np.array, W: np.array, b: np.array, activation: str) -> tuple:
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    # test for valid options
    assert activation == "sigmoid" or activation == "relu"
    
    # test for object type
    assert isinstance(A_prev, np.ndarray), "A_prev no es un np.ndarray"
    assert isinstance(W, np.ndarray), "W no es un np.ndarray"
    assert isinstance(b, np.ndarray), "b no es un np.ndarray"
    
    # test for shape
    assert A_prev.ndim == 2, "A_prev no es un rank 2 array"
    assert W.ndim == 2, "W no es un rank 2 array"
    assert b.ndim == 2, "b no es un rank 2 array"
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)        
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache

###   2.3) Combinación de 2.1 y 2.2  ###
def L_model_forward(X: np.array, parameters: dict) -> tuple:
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- activation value from the output (last) layer
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2 #  number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(
            A_prev, parameters['W' + str(l)], parameters['b' + str(l)], 'relu'
            )
        caches.append(cache)        
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL , cache = linear_activation_forward(
        A, parameters['W' + str(L)], parameters['b' + str(L)], 'sigmoid'
        )
    caches.append(cache)    
          
    return AL, caches



### 3) Función de costo  ###
### Para implementar la función de costo necesitamos un test set para comparar el valor de las predicciones
def compute_cost(AL: np.array, Y: np.array) -> float:
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))/m    
    
    cost = np.squeeze(cost) #  To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


###    4) Implementación de backward propagation   ###
###    4.1) Completa la capa LINEAR hacia atrás    ###
###                 Linear-Backward                ###

def linear_backward(dZ: np.array, cache: tuple) -> tuple:
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)    
    
    return dA_prev, dW, db


###    4.2) Completa la capa LINEAR-ACTIVATION hacia atrás    ###
###                 Linear-Activation-Backward                ###
def linear_activation_backward(dA: np.array, cache: tuple, activation: str) -> tuple:
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)        
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)        
    
    return dA_prev, dW, db

###    4.3) Implementanción de backward propagation L capas     ###
###                 Backward-Propagation                        ###
def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))    
    
    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
        dAL, current_cache, 'sigmoid'
        )
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp    
    
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l+1)], current_cache, 'relu'
            )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp        

    return grads


###  5) Actualización de parámetros   ###
def update_parameters(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    params -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]       

    return parameters


#
#
#
#                        PROGRAMA PRINCIPAL
#
#
#
#


if __name__ == "__main__":
    
    ###  1. Inicializacion  ###
    print("\n\n###  1. Inicializacion  ###\n")
    print("Test Case 1:\n")
    parameters = initialize_parameters(3,2,1)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    
    # print("\033[90m\nTest Case 2:\n")# le cambia el color
    print("\033[90m\nTest Case 2:\n")# le cambia el color
    parameters = initialize_parameters(4,3,2)

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    
    print("Test Case 1:\n")
    parameters = initialize_parameters_deep([5,4,3])

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


    print("\033[90m\nTest Case 2:\n")
    parameters = initialize_parameters_deep([4,3,2])

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    print("="*100, "\n\n")

    ## 2.1. Implementación de forward propagation ##

    n_0 = 3
    n_1 = 1
    n_2 = 1

    t_A = np.random.randn(3,2) # A0: nuestra muestra de training sets

    parameters = initialize_parameters(n_0, n_1, n_2)
    t_W1 = parameters["W1"]
    t_b1 = parameters["b1"]
    t_W2 = parameters["W2"]
    t_b2 = parameters["b2"]

    print(f"Input \"t_A\" shape: {t_A.shape}")
    print(f"Input \"t_W1\" shape: {t_W1.shape}")
    print(f"Input \"t_b1\" shape: {t_b1.shape}")

    assert(t_A.shape == (n_0, t_A.shape[1]))
    assert(t_W1.shape == (n_1, n_0))
    assert(t_b1.shape == (n_1, 1))

    # linear_forward no se implementa directamente
    t_Z, t_linear_cache = linear_forward(t_A, W=t_W1, b=t_b1)
    print("Z = " + str(t_Z))
    print("shape:", t_Z.shape)

    ### 2.2 Implementación de linear-activation forward ###
    m = 2
    n_0 = 3
    n_1 = 1
    n_2 = 1

    t_A_prev = np.random.randn(n_0, m) # A0: nuestra muestra de training sets

    parameters = initialize_parameters(n_0, n_1, n_2)
    t_W1 = parameters["W1"]
    t_b1 = parameters["b1"]
    t_W2 = parameters["W2"]
    t_b2 = parameters["b2"]
    
    print("\nForward propagation con linear and activation\n")
    
    print(f"Input \"t_A_prev\" shape: {t_A_prev.shape}")
    print(f"Input \"t_W1\" shape: {t_W1.shape}")
    print(f"Input \"t_b1\" shape: {t_b1.shape}")
    t_A, t_linear_activation_cache = linear_activation_forward(
        A_prev=t_A_prev, W=t_W1, b=t_b1, activation = "sigmoid")
    print("With sigmoid: A = " + str(t_A))

    t_A, t_linear_activation_cache = linear_activation_forward(
        A_prev=t_A_prev, W=t_W1, b=t_b1, activation = "relu")
    print("With ReLU: A = " + str(t_A)) #  Rectified Linear Unit (ReLU)


    ###   2.3)  ###
    ### modelo con 2 capas ocultas ###
    ### modelo con la capa output con función de activación sigmoid (Binary Classification) ###

    m = 4 #  4 muestras en el training set
    n_0 = 5 #  5 features en el training set
    L = 3 #  3 capas en la red neuronal
    n_1 = 4 #  4 neuronas en la capa 1
    n_2 = 3 #  3 neuronas en la capa 2
    n_3 = 1 #  1 neurona en la capa 3 # Clasificador Binario

    t_X = np.random.randn(n_0, m) # A0: nuestra muestra de training sets
    t_Y = np.random.randn(n_3, m) # Y: nuestra muestra de testing sets

    parametros = initialize_parameters_deep([n_0, n_1, n_2, n_3])

    t_AL, t_caches = L_model_forward(t_X, parametros)
    print("AL = " + str(t_AL))

    # ###   3) CALCULO DEL COSTO  ###
    t_cost = compute_cost(t_AL, t_Y)
    print("Cost: " + str(t_cost))

    ###   4) BACKWARD PROPAGATION   ###
    print("\nBackwards propagation\n")
    grads = L_model_backward(t_AL, t_Y, t_caches)

    print("dA0 = " + str(grads['dA0']))
    print("dA1 = " + str(grads['dA1']))
    print("dW1 = " + str(grads['dW1']))
    print("dW2 = " + str(grads['dW2']))
    print("db1 = " + str(grads['db1']))
    print("db2 = " + str(grads['db2']))


    ###  5) Actualización de parámetros   ###
    parametros = update_parameters(parametros, grads, learning_rate=0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))






    print("\nFinalización exitosa del programa\n")



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