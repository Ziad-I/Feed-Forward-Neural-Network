"""
    we only need one hidden layer and one output layer
    the output layer should be a layer that produces a 
    continuous output(linear). 
    i needed to add activation attribute to the layer class.

    I have a small question, is the A of the last layer the final output?
    I assumed this, correct me if i am wrong
"""

'''
read this: https://towardsdatascience.com/backpropagation-made-easy-e90a4d5ede55

Z = W*X + b
A = sigmoid(Z)
where:
    W is the weights of the current layer
    X is the output of previous layer (features if we are at first hidden layer)
    A is output of current layer (y_hat if we are in output layer)
    Note: we call X as A_prev since it is not always features

Note: we use number of samples (m) as we are using vectorized code 
instead of iterating over every sample

Input to the NN, X_train.shape = (number of features, m)
W.shape = (number of neurons in layer, inputs)
b.shape = (number of neurons in layer, 1)
Z.shape = (number of neurons in layer, m)
A.shape = Z.shape

all derivatives of something are same shape as that something
ex: dZ.shape = Z.shape
'''
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


class NeuralNetwork:
    def __init__(self, layers, epochs=100):
        '''
        Constructs a neural network with the given layers.

        Parameters:
        - layers (list): List of Layer objects representing the layers of the network.
        - epochs (int): number of epochs (iterations) for forward and backward passes
       '''
        self.layers = layers
        self.epochs = epochs

    def train(self, X_train, y_train):
        '''
        Train the neural network using gradient descent.

        Parameters:
        - X_train (numpy.ndarray): Input data of shape (input features, m).
        - y_train (numpy.ndarray): True labels of shape (output layer neurons, m).
        
        Returns:
        - None
        '''

        for _ in range(self.epochs):
            # the class structure forces me to do two separate loops

            # forward prop
            A_prev = X_train
            for i in range(len(self.layers)):
                A = self.layers[i].forward_propagation(A_prev)
                A_prev = A
            
            # backward prop
            dA = d_loss(y_train, A)

            for i in range(len(self.layers) - 1, -1, -1):
                dA_prev = self.layers[i].backward_propagation(dA)
                dA = dA_prev

    def predict(self, X_test):
        '''
        Make predictions using the trained neural network.

        Parameters:
        - X (numpy.ndarray): Input data of shape (input features, m).

        Returns:
        - predictions (numpy.ndarray): Predicted labels of shape.
        '''
        
        # just do forward prop
        A_prev = X_test
        for i in range(len(self.layers)):
            A = self.layers[i].forward_propagation(A_prev)
            A_prev = A
        return A
    
    def error(self, y_true, y_pred):
        """
        Compute the mean squared error (MSE) between the true and predicted values.

        Parameters:
        - y_true (array-like): Ground truth (true values).
        - y_pred (array-like): Predicted values.

        Returns:
        - float: Mean squared error between the true and predicted values.
        """
        return mean_squared_error(y_true, y_pred)

class Layer:
    '''
    This class represents a layer in a neural network.

    Attributes:
    - learning_rate (float): Learning rate for weight updates.
    - activation (callable): the activation function for this layer
    - d_activation (callable): the gradient of the activation
        function for this layer
    - W (numpy.ndarray): Weight matrix of shape (neurons, inputs).
    - b (numpy.ndarray): Bias matrix of shape (neurons, 1).
    - Z (numpy.ndarray): Computed Z values for this layer of shape (neurons, inputs)
    - A (numpy.ndarray): Layer output of shape same as Z
    - A_prev (numpy.ndarray): output of previous layer.

    Note:
     A and Z are stored to be used in backward propagation step

    to use:
    >>> layers = [
        Layer(inputs=4, neurons=3, activation="sigmoid", learning_rate=0.01),
        Layer(inputs=3, neurons=1, activation="linear", learning_rate=0.01)
        ]
    '''

    def __init__(self, inputs, neurons, activation="sigmoid", learning_rate=0.01):
        '''
        Constructs a layer that uses one of the defined activation functions
        for a neural network
        
        Parameters:
        - inputs (int): Number of input from previous layer (features if we are at first layer).
        - neurons (int): Number of neurons in the layer.
        - activation (string): the function name to be used as an activation function
        - learning_rate (float, optional): Learning rate for weight updates (default is 0.01).
        '''
        activation = activation.lower()
        self.activation = eval(activation)
        self.d_activation = eval("d_" + activation)
        self.learning_rate = learning_rate
        self.W = np.random.randn(neurons, inputs)
        self.b = np.zeros((neurons, 1))
        self.A_prev = None
        self.A = None
        self.Z = None

    def forward_propagation(self, A_prev):
        '''
        Perform a feedforward pass through the layer.

        Given the input activations from the previous layer `(A_prev)`,
        compute the weighted sum `(Z)` and the activated output `(a)` of the layer.

        Z = W*A_prev + b

        A = activation(Z)

        Parameters:
        - A_prev (numpy.ndarray): Input activations from the previous layer, 
            of shape (previous layer neurons, m).

        Returns
        - A (numpy.ndarray): Output of layer, of shape (neurons, m).
        '''
        self.A_prev = A_prev
        self.Z = np.dot(self.W, self.A_prev) + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward_propagation(self, dA):
        '''
        Perform backward propagation to update weights and biases and compute derivatives.
        
        Given the derivative of the cost function with respect to the layer's output (dA),
        compute the derivatives of the cost function with respect to the layer's parameters.
        Update the weights (W) and biases `(b)` using the computed derivatives.

        by doing the following steps:
        1. Calculate the derivative of the activation function with respect to Z `(dZ)`
        2. Compute the derivative of the cost function with respect to the weights `(dW)`
        3. Compute the derivative of the cost function with respect to the biases `(db)`
        4. Compute the derivative of the cost function with respect to the 
            input activations from the previous layer `(dA_prev)`
        5. Update the weights `(W)` and biases `(b)` using gradient descent

        Parameters:
        - dA (numpy.ndarray): derivative of the cost function 
            with respect to the layer's output, of shape (neurons, m).

        Returns:
        - dA_prev (numpy.ndarray): derivative of the cost function
            with respect to the input activations from the previous layer,
            of shape (previous layer neurons, m).
        '''
        # number of samples
        m = dA.shape[1]
        
        dZ = dA * self.d_activation(self.Z)
        dW = (1/m) * np.dot(dZ, self.A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return dA_prev



# Activation Functions
def linear(x):
    """
    Linear activation function.

    Args:
     - x (numpy.ndarray): Input to the activation function.

    Returns:
     - numpy.ndarray: Output of the activation function.
    """
    return x

def d_linear(x):
    """
    Derivative of the linear activation function.

    Args:
     - x (numpy.ndarray): Input to the activation function.

    Returns:
     - numpy.ndarray: Derivative of the linear activation function.
    """
    # The derivative of the linear function is always 1
    return np.ones_like(x)

def sigmoid(x):
    '''
    Compute the sigmoid activation function for a given input.

    Parameters:
    - x (numpy.ndarray): Input array.

    Returns:
    - numpy.ndarray: Output array after applying the sigmoid activation function.
    '''
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    '''
    Compute the derivative of the sigmoid activation function.

    Parameters:
    - x (numpy.ndarray): Input array.

    Returns:
    - numpy.ndarray: the derivative of the sigmoid function.
    '''
    s = sigmoid(x)
    return (1 - s) * s

# Loss Functions
def loss(y_true, y_pred):
    '''
    Compute the mean squared error (MSE) loss between the true and predicted values.

    Parameters:
    - y_true (numpy.ndarray): True values.
    - y_pred (numpy.ndarray): Predicted values.

    Returns:
    - float: Mean squared error loss.
    '''
    return np.mean(np.power(y_true-y_pred, 2))

def d_loss(y_true, y_pred):
    """
    Compute the derivative of the mean squared error (MSE) loss with respect to the predicted values.

    Parameters:
    - y_true (numpy.ndarray): True values.
    - y_pred (numpy.ndarray): Predicted values.

    Returns:
    - numpy.ndarray: Derivative of the mean squared error loss with respect to 'y_pred'.
    """
    return 2 * (y_pred - y_true) / len(y_true)
