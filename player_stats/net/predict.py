"""
NOTE! - currentlt testing the player stats NN with team statistics as dataset has not been created yet.
"""
import numpy as np
import pandas as pd
from net2 import Network
#from net import Network
from scipy.special import expit


# import_training_data = pd.read_csv("team_stats_train.csv")

# X = list(import_training_data.values[:,4] - import_training_data.values[:,5])

# y = list(import_training_data.values[:,2])

# training_data = zip(X,y)

# import_test_data = pd.read_csv("team_stats_test.csv")
# X_tst = list(import_test_data.values[:,4] - import_test_data.values[:,5])


training = pd.read_csv("players_1980.csv")
#print(len(training.values))

#X = list(training.values[:,5],training.values[:,6],training.values[:,7])
X = []

#X =np.array([training.values[0,5:8]])
l = 0
for l in range(len(training.values)):
	X.append([training.values[l,5:8]])

#print(X) 

y = []

i = 0
for i in range(len(training.values)):
	y.append([training.values[i,-1]])

#print(list(y))
#print(list(X))

n = Network(3,2,1,0.001)

epochs = 10


# for (training_input, training_output) in zip(X,y):
# 	print(training_input, training_output)

for e in range(epochs):
	for training_input, training_output in zip(list(X),list(y)):
		n.train(training_input,training_output)


n.feedforward([80,3000,94])
