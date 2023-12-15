# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from scipy import sparse
import mglearn
from sklearn.metrics import confusion_matrix

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
## Random Forest
#
def random_forest(features, target, Crit, nb_estim, Split, Depth):
    """
    To train the random forest classifier with features and target data
    :criterion='gini': function to measure the quality of a split. 
    Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
    :max_depth=None: maximum depth of the tree.
    :min_samples_split=2: minimum number of samples required to split an internal node.
    :max_features='auto': number of features to consider when looking for the best split.
    :max_leaf_nodes=None: Grow trees with max_leaf_nodes in best-first fashion. 
    Best nodes are defined as relative reduction in impurity. 
    If None then unlimited number of leaf nodes.
    :min_impurity_decrease=0: A node will be split if this split induces 
    a decrease of the impurity greater than or equal to this value.
    :min_samples_leaf=1: minimum number of samples required to be at a leaf node.
    :bootstrap=True: Whether bootstrap samples are used when building trees.
    :class_weight=None
    :min_weight_fraction_leaf: minimum weighted fraction of the sum total of weights 
    (of all the input samples) required to be at a leaf node.
    .n_estimators: number of trees in the forest
    :n_jobs=1
    :oob_score=False
    :random_state=None
    :verbose=0
    :warm_start=False
    return: trained random forest classifier
    """
    
    clf = RandomForestClassifier(criterion=Crit, max_depth=Depth, 
          min_samples_split=Split, n_estimators=nb_estim)
    clf.fit(features, target)
    return clf

print('Test Random Forest classifier')

crit = ['gini']
nb_Est = [25, 50, 100, 200, 500]
spl = [2]
depth = [10, 25, 50, 70, 100, 500]


param_grid = [{'criterion':crit, 'n_estimators':nb_Est, 'min_samples_split':spl, 
               'max_depth':depth}]
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3)
clf.fit(train_x, train_y)

print(clf.best_params_)
print(clf.best_score_)
print(clf.best_estimator_)
print(clf.grid_scores_)
scores = [score.mean_validation_score for score in clf.grid_scores_]
scores = np.array(scores).reshape(6, 5)

# plot the mean cross-validation scores
fig = plt.figure(figsize=(7,7))
fig = plt.figure(dpi=200)
mglearn.tools.heatmap(scores, xlabel='n_estimators', ylabel='max_depth', xticklabels=nb_Est,
yticklabels=depth, cmap="viridis")
fig.savefig('scores_RF.jpg')

# plot confusion matrix
class_y = clf.best_estimator_.predict(test_x)

fig = plt.figure(figsize=(17,17))
plt.xticks(rotation=60,fontsize=20)
plt.yticks(fontsize=20)
nom_pays = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian',
 'mexican', 'chinese', 'british', 'thai', 'vietnamese', 'cajun_creole',
 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']
mglearn.tools.heatmap(confusion_matrix(test_y, class_y), xlabel='Labels prédits',
ylabel='vrais labels', xticklabels=nom_pays, yticklabels=nom_pays,cmap=plt.cm.gray_r, fmt="%d")
plt.xlabel('Labels prédits',fontsize=20)
plt.ylabel('Vrais labels',fontsize=20)
plt.gca().invert_yaxis()
fig.savefig('M_confusion_RF.jpg')

