# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pylab as pl
import matplotlib.pyplot as plt
from scipy import sparse

Y = pd.read_csv("Y.csv")
Y = np.ravel(Y.values)
print(Y.shape)
X = sparse.load_npz("X.npz")
X_tfidf = sparse.load_npz("X_tfidf.npz")


### Algorithm implementation ###

# Split dataset into training and testing data
print('Separation des donnees entrainement/test')
train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8)

train_x_tfidf, test_x_tfidf, train_y, test_y = train_test_split(X_tfidf, Y, train_size=0.8)


## Régression logistique
#:penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
#Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers 
#support only l2 penalties.
#New in version 0.19: l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)
#
#:solver : str, {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default: ‘liblinear’.
#Algorithm to use in the optimization problem.
#For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster 
#for large ones.
#For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; 
#‘liblinear’ is limited to one-versus-rest schemes.
#‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ 
#and ‘saga’ handle L1 penalty.
#Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately 
#the same scale. You can preprocess the data with a scaler from sklearn.preprocessing.
#New in version 0.17: Stochastic Average Gradient descent solver.
#New in version 0.19: SAGA solver.
#Changed in version 0.20: Default will change from ‘liblinear’ to ‘lbfgs’ in 0.22.
#
#:max_iter : int, default: 100
#Useful only for the newton-cg, sag and lbfgs solvers. Maximum number of iterations taken 
#for the solvers to converge.:
#    
#:multi_class : str, {‘ovr’, ‘multinomial’, ‘auto’}, default: ‘ovr’
#If the option chosen is ‘ovr’, then a binary problem is fit for each label. 
#For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire 
#probability distribution, even when the data is binary. ‘multinomial’ is unavailable 
#when solver=’liblinear’. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, 
#and otherwise selects ‘multinomial’.
#
#New in version 0.18: Stochastic Average Gradient descent solver for ‘multinomial’ case.
#Changed in version 0.20: Default will change from ‘ovr’ to ‘auto’ in 0.22.

#:class_weight : dict or ‘balanced’, default: None
#Weights associated with classes in the form {class_label: weight}. 
#If not given, all classes are supposed to have weight one.
#The “balanced” mode uses the values of y to automatically adjust weights 
#inversely proportional to class frequencies in the input data as 
#n_samples / (n_classes * np.bincount(y)).
#Note that these weights will be multiplied with sample_weight (passed through the fit method) 
#if sample_weight is specified.
#New in version 0.17: class_weight=’balanced’


#print('Logistic regression classifier')

clf1 = LR(multi_class='multinomial', solver='newton-cg')
print ('X')
print(cross_val_score(clf1, X, Y, cv=3))
clf1.fit(train_x, train_y)
class_y = clf1.predict(test_x)
print(clf1.score(train_x, train_y))
print(accuracy_score(test_y, class_y))

print('X_tfidf')
print(cross_val_score(clf1, X_tfidf, Y, cv=3))
clf1.fit(train_x_tfidf, train_y)
class_y = clf1.predict(test_x_tfidf)
print(clf1.score(train_x_tfidf, train_y))
print(accuracy_score(test_y, class_y))

clf2 = LRCV(cv=4, solver='newton-cg', multi_class='multinomial')
print ('X_cv')
#print(cross_val_score(clf2, X, Y, cv=3))
clf2.fit(train_x, train_y)
class_y = clf2.predict(test_x)
print(clf2.score(train_x, train_y))
print(accuracy_score(test_y, class_y))

print('X_tfidf_cv')
#print(cross_val_score(clf2, X_tfidf, Y, cv=3))
clf2.fit(train_x_tfidf, train_y)
class_y = clf2.predict(test_x_tfidf)
print(clf2.score(train_x_tfidf, train_y))
print(accuracy_score(test_y, class_y))