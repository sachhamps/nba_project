"""
net2.py
~~~~~~~~~~
Feed-forward neural network that will use SGD and backpropagation to
update weights and biases. This FNN will initially be created to predict
the PPG for a selected player. This NN will be expanded, allowing it to
predict more stats such as APG, RPG etc. The final goal will be predictions for
teams and players.

Heavily based on network.py and 'Make you own Neural Network' book, but changes 
will be made as this is a basic/first draft attempt at a neural network.


FIX BUGS RELATING TO DIMENSIONALITY!!!


"""
#import random
from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt

class Network(object):

	def __init__(self, inputsize, hiddensize, outputsize, learningrate):

		self.input_nodes = inputsize
		self.hidden_nodes = hiddensize
		self.output_nodes = outputsize
		self.eta = learningrate


		#initialising weights between input and hidden layer
		self.weights_1 = np.random.randn(self.input_nodes, self.hidden_nodes)

		#initialising weights between hidden layer and output layer
		self.weights_2 = np.random.randn(self.hidden_nodes, self.output_nodes)

	def feedforward(self, inputs):

		a = np.array(inputs,ndmin=2,dtype=np.float64).T
		#propogate the the  activation through the neural network

		hidden_layer = np.dot(self.weights_1.T,a)
		#z values for the hidden layer in sigmoid function

		hidden_activation = sigmoid(hidden_layer)
		output_layer = np.dot(self.weights_2.T,hidden_layer)
		#z values for the output layer in sigmoid function 

		output_activation = sigmoid(output_layer)

		#print(output_layer)

		#print(output_activation)
		return output_activation

	def cost_derivative(self, output_activation,y):
		return (output_activation - y)


	def train(self, x_prime, y_prime):

		#eliminate this when input becomes array rather than list
		x = np.array(x_prime, ndmin=2,dtype=np.float64).T
		y = np.array(y_prime, ndmin=2,dtype=np.float64).T
		#print(x)
		#print(y)

		#feedforward
		hidden_activation = sigmoid(np.dot(self.weights_1.T,x))
		output_activation = sigmoid(np.dot(self.weights_2.T, hidden_activation))


		
		#calculate the error of the output layer
		delta = self.cost_derivative(output_activation,y)
		delta_hidden = np.dot(self.weights_2,delta)

		#update the weights for each layer
		#print("weights intermediate is: {0}".format(delta * output_activation * (1 - output_activation)))
		#print("hidden: {0}".format(hidden_activation.T))
		#print("weights: {0}".format(self.weights_2))


		self.weights_2 += self.eta * np.dot((delta * output_activation * (1 - output_activation)), hidden_activation.T).T
		self.weights_1 += self.eta * np.dot((delta_hidden * hidden_activation * (1 - hidden_activation)), x.T).T
		#print("weights 2 = " + str(self.weights_2))
		#print("weights 1 = "  + str(self.weights_1))




def sigmoid(x):
    return (expit(x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))



#n = Network(3,2,1,0.3)
#n.train([1.0, 0.5,-1.5], [25])
#n.feedforward([0.3,0.5,0.6])

