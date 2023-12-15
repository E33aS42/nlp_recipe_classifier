# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
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

    
# XGboost
DM_train = xgb.DMatrix(data = train_x, label = train_y)  
DM_test =  xgb.DMatrix(data = test_x, label = test_y)

def Extr_grad_boosting(X_train, y_train, gbm_param_grid):
    '''
    .Learning rate/eta- governs how quickly the model fits the residual error using additional base learners. 
    If it is a smaller learning rate, it will need more boosting rounds, hence more time, 
    to achieve the same reduction in residual error as one with larger learning rate. Typically, it lies between 0.01 – 0.3
    
    The three hyperparameters below are regularization hyperparameters.
    .gamma: min loss reduction to create new tree split. default = 0 means no regularization.
    .lambda: L2 reg on leaf weights. Equivalent to Ridge regression.
    .alpha: L1 reg on leaf weights. Equivalent to Lasso regression.
    
    .max_depth: max depth per tree. This controls how deep our tree can grow. 
    The Larger the depth, more complex the model will be and higher chances of overfitting. 
    Larger data sets require deep trees to learn the rules from data. Default = 6.
    
    .subsample: % samples used per tree. This is the fraction of the total training set 
    that can be used in any boosting round. Low value may lead to underfitting issues. 
    A very high value can cause over-fitting problems.
    
    .colsample_bytree: % features used per tree. This is the fraction of the number of columns 
    that we can use in any boosting round. A smaller value is an additional regularization 
    and a larger value may be cause overfitting issues.
    
    .n_estimators: number of estimators (base learners). This is the number of boosting rounds.
    '''
    
#gbm_param_grid = {
#     'colsample_bytree': np.linspace(0.5, 0.9, 5),
#     'n_estimators':[100, 200],
#     'max_depth': [10, 15, 20, 25]
#}

    
#xg_reg.fit(train_x, train_y)
#preds = xg_reg.predict(test_x)
#rmse = np.sqrt(mean_squared_error(test_y, preds))
#print(xg_reg.score(train_x, train_y))
#print(cross_val_score(xg_reg, train_x, train_y, cv=5))


#ex2
# specify parameters via map
#param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
#num_round = 2
#bst = xgb.train(param, DM_train, num_round)
## make prediction
#preds = bst.predict(DM_test)
#print(bst.score(train_x, train_y))
#print(cross_val_score(bst, train_x, train_y, cv=3))
#print(cross_val_score(bst, test_x, test_y, cv=3))

#ex3
#max_depth (int) – Maximum tree depth for base learners.
#learning_rate (float) – Boosting learning rate (xgb’s “eta”)
#n_estimators (int) – Number of boosted trees to fit.
#silent (boolean) – Whether to print messages while running boosting.
#objective (string or callable) – Specify the learning task and the corresponding learning objective or a custom objective function to be used (see note below).
#booster (string) – Specify which booster to use: gbtree, gblinear or dart.
#nthread (int) – Number of parallel threads used to run xgboost. (Deprecated, please use n_jobs)
#n_jobs (int) – Number of parallel threads used to run xgboost. (replaces nthread)
#gamma (float) – Minimum loss reduction required to make a further partition on a leaf node of the tree.
#min_child_weight (int) – Minimum sum of instance weight(hessian) needed in a child.
#max_delta_step (int) – Maximum delta step we allow each tree’s weight estimation to be.
#subsample (float) – Subsample ratio of the training instance.
#colsample_bytree (float) – Subsample ratio of columns when constructing each tree.
#colsample_bylevel (float) – Subsample ratio of columns for each split, in each level.
#reg_alpha (float (xgb's alpha)) – L1 regularization term on weights
#reg_lambda (float (xgb's lambda)) – L2 regularization term on weights
#scale_pos_weight (float) – Balancing of positive and negative weights.
#base_score – The initial prediction score of all instances, global bias.
#seed (int) – Random number seed. (Deprecated, please use random_state)
#random_state (int) – Random number seed. (replaces seed)
#missing (float, optional) – Value in the data which needs to be present as a missing value. If None, defaults to np.nan.
#importance_type (string, default "gain") – The feature importance type for the feature_importances_ property: either “gain”, “weight”, “cover”, “total_gain” or “total_cover”.
#**kwargs (dict, optional) –

   
    

param_grid = [{'max_depth': [5], 'n_estimators':[100], 'learning_rate':[0.05, 0.1, 0.2, 0.25], 'n_jobs':[4]}]
svc = xgb.XGBClassifier()
clf = GridSearchCV(svc, param_grid, cv=4)
clf.fit(train_x, train_y)
print(clf.best_params_)
print(clf.best_score_)
     
#gbm1 = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.05)
#print(cross_val_score(gbm1, X, Y, cv=4))   
#clf.fit(train_x, train_y)
#gbm2 = xgb.XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.05)
#gbm2.fit(train_x, train_y)
#predictions = gbm2.predict(test_x)
#print(gbm2.score(train_x, test_y))





