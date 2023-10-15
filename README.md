# Modelo de Deep Learning para Clasificación Binaria

1. Dataset: Titanic
2. Modelo 1: Two Layer Neural Network
3. Modelo 2: Multi Layer Neural Network (L=5)
4. Cantidad de features = 9
5. Cantidad de muestas en el training set = 478
6. Cantidad de muestras en el testing set = 236


## Resumen Modelo 1: 2LNN

1. hidden layer: 9 neuronas, activation function ReLU
2. output layer: 1 neurona, activation function sigmoide

![output_modelo_1](https://github.com/jackonedev/deep_neural_networks/blob/main/images/2LNN.png?raw=true)

## Resumen Modelo 2: 5LNN

1. hidden layer 1: 9 neuronas, activation function ReLU
2. hidden layer 2: 9 neuronas, activation function ReLU
3. hidden layer 3: 5 neuronas, activation function ReLU
4. hidden layer 4: 3 neuronas, activation function ReLU
5. output layer: 1 neurona, activation function sigmoide

![output_modelo_2](https://github.com/jackonedev/deep_neural_networks/blob/main/images/5LNN.png?raw=true)

# Conclusiones

Ambos modelos tienen sus pros y sus contras. Para empezar el modelo 2LNN tuvo mejor score con datos nuevos que con los datos de entrenamiento. A su vez, alcanza el óptimo en su gradiente a un costo computacional mucho menor.

El modelo 5LNN es el resultado de varias experimentaciones en el diseño de su arquitectura. Llegando a ser este el modelo más performante. Aún así, requiere 10 veces más cantidad de iteraciones que el modelo 2LNN y la cantidad de cálculos es exponencialmente mayor. 

Aún así, entre un modelo y otro hay una diferencia mayor al 5% en su score alcanzado en el testing set. Considérese significativo.