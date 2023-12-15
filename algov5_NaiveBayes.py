# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
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
train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.7)
train_x_tfidf, test_x_tfidf, train_y, test_y = train_test_split(X_tfidf, Y, train_size=0.7)
#

print('Multiclass Naive Bayes classifier')

#:alpha : float, optional (default=1.0)
#Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
#:fit_prior : boolean, optional (default=True)
#Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
#:class_prior : array-like, size (n_classes,), optional (default=None)
#Prior probabilities of the classes. If specified the priors are not adjusted according to the data.


clf1 = NB(alpha=1.0, class_prior=None, fit_prior=True)
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

