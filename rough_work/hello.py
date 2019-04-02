#quick intro to ml in python,  using a decision tree to classify and predict
#apples or oranges based on 2 features. 1 = apples, 0 = oranges // 1 = bumpy, 0 = smooth

from sklearn import tree
import numpy as np

features = [[140, 1], [130,1], [150,0],[170,0]]
labels = [0,0,1,1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[141,0]]))
