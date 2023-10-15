from originals.dnn_app_utils_v3 import initialize_parameters_deep, L_model_forward, L_model_backward, predict
import numpy as np
import pandas as pd
from deep_neural_network import initialize_parameters, linear_activation_forward, linear_activation_backward, compute_cost, update_parameters
import matplotlib.pyplot as plt
from basics import normalize_rows, softmax


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

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)    
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
    
        # Compute cost.
        cost = compute_cost(AL, Y)        
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)        
                
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

# 
# 
# 
#                      PROGRAMA PRINCIPAL
# 
# 
# 
# 

if __name__ == "__main__":
    
    X, Y, X_test, Y_test = feed_exploration()

    # Reshape the training and test examples (innecesario en este caso)
    train_x_flatten = X.reshape(X.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = X_test.reshape(X_test.shape[0], -1).T

    train_x_flatten = np.array(train_x_flatten, dtype=np.float64)
    test_x_flatten = np.array(test_x_flatten, dtype=np.float64)
    
    # train_x_flatten = normalize_rows(train_x_flatten)
    # test_x_flatten = normalize_rows(test_x_flatten)
    
    # train_x_flatten = softmax(train_x_flatten)
    # test_x_flatten = softmax(test_x_flatten)
    

    print ("train_x's shape: " + str(X.shape))
    print ("test_x's shape: " + str(X_test.shape))
    
    ## HIPERPARAMETROS DEL MODELO
    
    n_x = X.shape[1] # numero de input features
    n_1 = 9
    n_2 = 9
    n_3 = 5 
    n_4 = 3 
    n_5 = 1
    
    layers_dims = [n_x, n_1, n_2, n_3, n_4, n_5] # MODELO DE 5 CAPAS
    # learning_rate = 0.0075
    learning_rate = 0.075
    num_iterations = 20000
    # num_iterations = 12000   normalizado # ahorra iteraciones pero el resultado es un poco peor
    # num_iterations = 1000 # softmax   # converge muy rápido y el score es malo
    
    ## ENTRENAMIENTO DEL MODELO
    ## primero lo probamos con pocas iteraciones
    # luego lo entrenamos con muchas iteraciones
    # de ese modo podemos regular el learning rate
    
    parameters, costs = L_layer_model(train_x_flatten, Y, layers_dims, num_iterations = num_iterations, print_cost = False)

    print("Cost after first iteration: " + str(costs[0]))
    
    # plot_costs(costs, learning_rate)
    
    pred_train = predict(train_x_flatten, Y, parameters)
    
    pred_test = predict(test_x_flatten, Y_test, parameters)
    