#taking data from the 2011 NBA Season - the last season present in the NBA_train dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd

training_data = pd.read_csv("NBA_train.csv")
#print(training_data)

print(np.array(training_data.values[805:835,3]))

X = np.matrix([training_data.values[805:835,3], training_data.values[805:835,4] - training_data.values[805:835,5]]) #wins and net points
X = X.T
y = np.array(training_data.values[805:835,2]) #1 if they made the playoffs, 0 if they did not make the playoffs
y = y.astype('int')
#plt.scatter(X[0],y)
#plt.show()


clf = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', solver='sgd', alpha=0.0001, batch_size='auto', learning_rate='constant',
					learning_rate_init=0.001, power_t=0.5, max_iter=10000, shuffle=True, random_state=None, tol=0.00000001, verbose=True, warm_start=False, 
					momentum=0.8, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1)
clf.fit(X,y)

test_data = pd.read_csv('NBA_test.csv')
X_tst =np.matrix([test_data.values[:,3], test_data.values[:,4] - test_data.values[:,5]])
X_tst = X_tst.T

output = clf.predict(X_tst)
y_actual = np.array(test_data.values[:,2])
y_actual = y_actual.astype('int')
count = 0

for i in range (len(output)):
	if output[i] == y_actual[i]:
		count = count + 1

print("final score:", count/len(y_actual))