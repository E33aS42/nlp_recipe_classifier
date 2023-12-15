# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import statsmodels.api as sm
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
train_x, test_x, train_y, test_y = train_test_split(X, Y,
                                                        train_size=0.8)
#

### Plot of the training and testing error over iteration time
def diagnos(X_test, Y_test, gb, params):
#     params is the dictionary of gradient boosting parameters
#    test_score = np.zeros((params['loss'], params['n_estimators'],
#                           params['learning_rate'], params['Depth'],), 
#                           dtype=np.float64)
    test_score = np.zeros((params,), dtype=np.float64)    
    
    for i, y_pred in enumerate (gb.staged_decision_function(X_test)):
        test_score[i] = gb.loss_(Y_test, y_pred)
    
#    pl.figure(figsize=12, 9)
    pl.figure()
    pl.plot(np.arange(params) + 1, gb.train_score_, 'b-', label='Training Set Deviance')
    pl.plot(np.arange(params) + 1, test_score, 'r-', label='Test Set Deviance')
    pl.legend(loc='upper right')
    pl.xlabel('Boosting Iterations')
    pl.ylabel('Deviance')
    pl.show()
    

# Gradient boosting classifier
                
def gradient_boosting(features, target, lossf, nb_estim, alpha, Depth):            
    '''GradientBoostingClassifier
    loss='deviance': loss function
    'ls': least square
    'deviance': logistic regression
    'exponential': Adaboost
    learning_rate=0.1: often inside the [0.03,0.2] window
    n_estimators=100:
    subsample=1.0
    criterion='friedman_mse' 
    min_samples_split=2: 
    min_samples_leaf=1
    min_weight_fraction_leaf=0.0
    max_depth=3: depth of a tree, usually low between 3 and 8
    min_impurity_decrease=0.0
    min_impurity_split=None
    init=None
    random_state=None
    max_features=None
    verbose=0
    max_leaf_nodes=None
    warm_start=False
    presort='auto'
    validation_fraction=0.1
    n_iter_no_change=None
    tol=0.0001
    '''                
    clf = GradientBoostingClassifier(loss=lossf, learning_rate=alpha, 
                                     n_estimators=nb_estim, max_depth=Depth)  
    clf.fit(features, target)
    return clf


#
print('Test Gradient Boosting classifier')
    
loss = ['deviance']#, 'exponential']
nb_Est = [200]
#alpha = [0.03, 0.1, 0.2]
alpha =[0.03]
#depth = [3, 8, 20]
depth = [3]
    
for i in loss:
    for j in nb_Est:
        for k in alpha:
            for m in depth:
                print('loss: %s, nb_Est: %s, alpha: %s, depth: %s.' %(i, j, k, m))
                classifier = gradient_boosting(train_x, train_y,i,j,k,m)
                class_y = classifier.predict(test_x)
                print(accuracy_score(test_y, class_y))
                print(classifier.score(train_x, train_y))
                print(cross_val_score(classifier, train_x, train_y, cv=3))
                print(cross_val_score(classifier, test_x, test_y, cv=3))
    
                params = {'loss' : i, 'n_estimators' : j, 'learning_rate' : k,
                          'Depth' : m}
                diagnos(test_x, test_y, classifier, j)
    

