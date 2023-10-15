from originals.dnn_app_utils_v3 import predict
import numpy as np
import pandas as pd
from deep_neural_network import initialize_parameters, linear_activation_forward, linear_activation_backward, compute_cost, update_parameters
import matplotlib.pyplot as plt

####   Binary Classification Model | two layer model  ####

def feed_exploration():

    # Importing the dataset
    X = pd.read_csv('X_train.csv')
    X = X.set_index(X.columns[0])
    X.index.name = None
    Y = pd.read_csv('y_train.csv')
    Y = Y.set_index(Y.columns[0])
    Y.index.name = None

    X_test = pd.read_csv('X_test.csv')
    X_test = X_test.set_index(X_test.columns[0])
    X_test.index.name = None
    Y_test = pd.read_csv('y_test.csv')
    Y_test = Y_test.set_index(Y_test.columns[0])
    Y_test.index.name = None


    X = X.to_numpy()#.reshape(X.shape[1], X.shape[0])
    Y = Y.to_numpy().reshape(Y.shape[1], Y.shape[0])
    X_test = X_test.to_numpy()
    Y_test = Y_test.to_numpy().reshape(Y_test.shape[1], Y_test.shape[0])
    
    # Explore your dataset 
    m_train = X.shape[0]
    num_px = X.shape[1]
    m_test = X_test.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each training set is of size: (" + str(X.shape[1:]) + ")" )
    print ("train_x_orig shape: " + str(X.shape))
    print ("train_y shape: " + str(Y.shape))
    print ("test_x_orig shape: " + str(X_test.shape))
    print ("test_y shape: " + str(Y_test.shape))

    return X, Y, X_test, Y_test


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')
 
        # Compute cost
        cost = compute_cost(A2, Y)        
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


if __name__ == "__main__":
    
    X, Y, X_test, Y_test = feed_exploration()

    # Reshape the training and test examples (innecesario en este caso)
    train_x_flatten = X.reshape(X.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = X_test.reshape(X_test.shape[0], -1).T

    train_x_flatten = np.array(train_x_flatten, dtype=np.float64)
    test_x_flatten = np.array(test_x_flatten, dtype=np.float64)

    print ("train_x's shape: " + str(X.shape))
    print ("test_x's shape: " + str(X_test.shape))
    
    train_x_flatten
    
    ###  Hiperparámetros del modelo  ###
    n_x = X.shape[1] # numero de input features
    n_h = 9# X.shape[1] # numero de neuronas en la capa oculta
    n_y = 1 # numero de neuronas en la capa de salida
    layers_dims = (n_x, n_h, n_y)
    # learning_rate = 0.0075
    learning_rate = 0.1
    num_iterations = 1500
    
    
    ### Entremos el modelo ###
    parameters, costs = two_layer_model(
        train_x_flatten, Y, 
        layers_dims = layers_dims, 
        num_iterations = num_iterations,
        learning_rate = learning_rate,
        print_cost=False
        )
    
    
    print("Cost after first iteration: " + str(costs[0]))
    
    plot_costs(costs, learning_rate)
    
    print("Train score:")
    predictions_train = predict(train_x_flatten, Y, parameters)
    
    print("Test score:")
    predictions_test = predict(test_x_flatten, Y_test, parameters)
    
    print("programa finalizado exitosamente")

