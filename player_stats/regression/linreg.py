"""
linreg.py
~~~~~~~~~~

Linear regression of a players stats over a period of seasons,
this will be used to caluculate a prediction of statlines for players
in the 2018-19 seasons - Currently a temporary dataset is being used. This
dataset uses players stats per season over their careers. The real dataset
will contain the stats of every game of a player over the previous 3 seasons
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial.distance import cdist
from numpy.linalg import inv
from sklearn.kernel_ridge import KernelRidge





data_set = pd.read_excel("playerDataSet.xlsx")
#jokic = pd.read_csv("/players/jokic.csv")
#(249,31)
data_array = np.array(data_set)
#print(data_array)
#all values representing the player_id column
player_ids = data_array[:,0]
#print(player_ids)
season_no = data_array[:,1]
points_per_game = data_array[:,30]
assists_per_game = data_array[:,25]
trb_per_game = data_array[:,24]
#print(season_no, points_per_game)
print(data_array)

#plot season number vs points per game
def extract_ppg(player_id):
	season = []
	ppg = []
	for i in range(player_ids.size):
		if(player_id == player_ids[i]):
			season.append(season_no[i])
			ppg.append(points_per_game[i])

	return season, ppg


#simple regression - make it a class and add uncertainty!!
def estimate_coefs(X, y):
	X = np.array(X, dtype=np.float64)
	y = np.array(y, dtype=np.float64)
	mu_x = np.mean(X)
	mu_y = np.mean(y)
	SS_x_y = np.sum(y*X) - len(y)*mu_x*mu_y
	SS_x_x = np.sum(X*X) - len(y) * (mu_x * mu_x)
	w_1 = SS_x_y / SS_x_x
	w_0 = mu_y - w_1 * mu_x

	return w_0, w_1


def plt_regression(X,y,w):
	X = np.array(X, dtype=np.float64)
	y = np.array(y, dtype=np.float64)
	plt.scatter(X,y)
	p = w[0] + w[1] * X
	plt.plot(X,p,color='b',alpha=0.5)


def prediction(X, w):
	X = np.array(X, dtype=np.float64)
	nxt_season = len(X) + 1
	pred = w[0] + nxt_season * w[1]
	print("next season ppg: {0}".format(pred))


#X,y = extract_ppg(8)
#w = estimate_coefs(X,y)
#plt_regression(X,y,w)
# prediction(X, w)

# to do next, add uncertainty to the current model, put the model in a class,
# then produce OLS for multiple features so statlines can be produced (PTS, REB, AST) and put
# that into a class - might not work
# then begin non-linear regression for this dataset before moving onto current nba players








"""
TensorFlow Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~

The code below is the tensorflow implementation of the model above (used for practice)
"""




learning_rate = 0.01
epochs = 2000
s, p = extract_ppg(3)

train_X = np.array(s, dtype=np.float32)
train_Y = np.array(p, dtype=np.float32)

n_samples = train_X.shape[0]

X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(np.random.randn, "weight")
b = tf.Variable(np.random.randn, "bias")

pred = tf.add(tf.multiply(X,W),b)

#mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2)) / (2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)

# 	for epoch in range(epochs):
# 		for(x,y) in zip(train_X, train_Y):
# 			sess.run(optimizer, feed_dict={X:x, Y:y})

# 		if (epoch+1) % 50 == 0:
# 			c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
# 			print("Epoch: ",epoch+1, "cost={0}".format(c), \
# 			"W=",sess.run(W), "b=",sess.run(b))

# 	print("Optimized!")

# 	plt.scatter(train_X, train_Y)
# 	plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line',color='r')
# 	plt.legend()
# 	plt.show()










"""
linreg with mulitple features
~~~~~~~~~~~~~~~~~~~~~~~~

multiple dimension linear regression will be extended for the following features:

X = season, eFG%, TS%
y = PPG
"""
n_seasons = np.array(season_no, dtype=np.float32)
n_ppg = np.array(points_per_game, dtype=np.float32)
n_apg = np.array(assists_per_game, dtype=np.float32)
t_rpg = np.array(trb_per_game, dtype=np.float32)
temp = np.column_stack((n_seasons, n_apg, n_ppg))
#print(temp)











"""
Kernel Ridge Regression
~~~~~~~~~~~~~~~~~~~~~~~~

note: build own kernel regressor rather than using sklearn
"""

train_X = train_X.reshape(-1,1)
train_Y = train_Y.reshape(-1,1)
clf = KernelRidge(alpha=0.5, kernel='rbf', gamma=0.05, degree=3, coef0=1, kernel_params=None)
clf.fit(train_X,train_Y)

y_kr = clf.predict(train_X)
plt.scatter(train_X, train_Y,color='r',alpha=0.5)
plt.plot(train_X, y_kr)
plt.show()

#y_kr2 = clf.predict(16)
#print(y_kr2)








"""
Gaussian Process from Machine Learining Coursework, could be implementing incorrectly!
"""

lmda = 1

def RBF(xi,xj,gamma):
	return np.exp(-gamma * cdist(xi,xj))


train_X = np.array(s, dtype=np.float32)
train_Y = np.array(p, dtype=np.float32)
w = estimate_coefs(train_X,train_Y)
w = np.array(w, dtype=np.float32)
train_X = train_X.reshape(-1,1)
#K = RBF(train_X,train_X,1)
#print(K.shape, w.shape)


#compute the predictive posterior, x = x* 
def predictive_posterior(X,x,Y):
    K = RBF(X,X,0.5)
    K_2 = RBF(X,x,0.5).T
    K_3 = RBF(x,x,0.5)
    mean = K_2 @ inv((K) + ((0.3**2) * np.eye(len(Y)))) @ Y
    sigma = K_3 - K_2 @ inv((K)+((0.3**2) * np.eye(len(Y)))) @ K_2.T
    
    return mean,sigma

def plot_pp(mean, sigma,n):
    #plotting the space of the functions with updated mean values and variance
    plt.scatter(train_X,train_Y)
    plt.plot(train_X,mean,color='r')
    plt.show()










"""
Miscellaneous
"""

def weight_vector(Xmat, Y, alpha):
    # catch cases where there is only 1 feature
    if type(Xmat) != np.ndarray:
        Xmat = np.array(Xmat)
        Xmat = Xmat.reshape((len(Xmat),1))
    num_features = Xmat.shape[1]
    numerator = np.dot(Xmat.T, Xmat) + alpha*np.identity(num_features)
    numerator = np.linalg.inv(numerator)
    denominator = np.dot(Xmat.T, Y)
    w = np.dot(numerator, denominator)
    y_hat = np.dot(Xmat, w)
    return w, y_hat

def plotmatrix(Matrix):
  r, c = Matrix.shape
  fig = plt.figure()
  plotID = 1
  for i in range(c):
    for j in range(c):
      ax = fig.add_subplot( c, c, plotID )
      ax.scatter( Matrix[:,i], Matrix[:,j] )
      plotID += 1
  plt.show()
  



