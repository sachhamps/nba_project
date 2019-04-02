"""
net.py
~~~~~~~~~~
Feed-forward neural network that will use SGD and backpropagation to
update weights and biases. This FNN will initially be created to predict
the PPG for a selected player. This NN will be expanded, allowing it to
predict more stats such as APG, RPG etc. The final goal will be predictions for
teams and players.

THIS FNN WILL BE CHANGED INORDER TO PREDICT THE OUTCOME
OF TEAM STATS - net2.py is an attempt to predict player stats.

Heavily based on network.py, but changes will be made as this is a basic/first draft
attempt at a neural network
"""
# Standard library
#import random
from scipy.special import expit

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, input_size, hidden_size, output_size):

        #initialising layers
        self.input_layer = input_size
        self.hidden_layer = hidden_size
        self.outpt_layer = output_size

        #creating a list of neuron layer sizes
        self.sizes = [input_size, hidden_size, output_size]

        #initialising weights and biases randomly 
        self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1],self.sizes[1:])]
        self.biases = [np.random.randn(y,1) for y in self.sizes[1:]]
        

    #progagates activation - assumes a is (n,1) ndarray not a (n,) vector
    def feed_forward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def GD(self, training_data, eta, epochs, test_data=None):

        for j in range(epochs):
            updated_b = [np.zeros(b.shape) for b in self.biases]
            updated_w = [np.zeros(w.shape) for w in self.weights]

            for x,y in training_data:
                delta_b, delta_w = self.back_prop(x,y)
                updated_b = [ub + db for ub,db in zip(updated_b,delta_b)]
                updated_w = [uw + dw for uw,dw in zip(updated_w,delta_w)]

            self.weights = [w - (eta * uw) for w,uw in zip(self.weights, updated_w)]
            self.biases =  [b - (eta * ub) for b,ub in zip(self.biases, updated_b)]

            print("Epoch {0} complete".format(j))


## FIX BACKPROP FUNCTION - DIMENSIONALITY ISSUE!!!
    def back_prop(self, x ,y):
        temp_b = [np.zeros(b.shape) for b in self.biases]
        temp_w = [np.zeros(w.shape) for w in self.weights]

        #feedforword
        activation = x
        print(activation)
        activations = [x] #store the activations layers by layer, starting with the first
        print("here")
        zs = [] #list of z values layer by layer

        #iterate through every bias and weight
        for b,w in zip(self.biases,self.weights):
            
            z = (activation * w) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)


        #backpropagate the error
        delta = self.cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])
        temp_b[-1] = delta #bias error of the last layer
        temp_w[-1] = np.dot(activations[-2].T,delta) #weight errors of the last layer

        for l in range(2,len(self.sizes)):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_prime(z)
            temp_b[-l] = delta
            temp_w[-l] = np.dot(activations[-l-1].T, delta)
        return(temp_b,temp_w)




    def cost_derivative(self, final_activation, y):
        return (final_activation - y)


def sigmoid(x):
    return (expit(x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))