import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier

import utils

# Load data
train_input, train_output = utils.loadTrainingData(remove_id=True, remove_capuchon_insertion=True)

# Feature selection
# print(train_input.shape)

# selector = VarianceThreshold(threshold=0.8)
# train_input = pd.DataFrame(selector.fit_transform(train_input))

# train_input_ = pd.DataFrame(SelectKBest(f_classif, k=4).fit_transform(train_input, train_output))

# print(train_input_.shape)

# Create model
clf = GaussianNB()
# clf = KNeighborsClassifier(n_neighbors=3) # weights="distance"
# clf = ComplementNB()

# Grid search
# params_NB = {'var_smoothing': np.logspace(0, -15, num=100)}
# # print(np.logspace(0, -9, num=10))
# gs_NB = GridSearchCV(estimator=clf,
#                      param_grid=params_NB,
#                      # cv=cv_method,   # use any cross validation technique
#                      verbose=1,
#                      scoring='roc_auc')
# gs_NB.fit(train_input, train_output)

# print(gs_NB.best_params_)

# roc_auc -> 6*10**-7
# precision -> 1
# recall -> 1
# f1 -> 1

# Prediction
# y_pred = cross_val_predict(clf, train_input, train_output['result'], cv=5)
# print(y_pred)


# Evaluate model
# utils.modelEvaluation(clf, train_input, train_output, balance_classes=True, model_name="Test Classifier", fig_name="test")
