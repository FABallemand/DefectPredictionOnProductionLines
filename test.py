import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import utils

# Load data
train_input, train_output = utils.loadTrainingData(remove_id=True, remove_capuchon_insertion=True)

# Create model
# clf = GaussianNB()
clf = KNeighborsClassifier(n_neighbors=3) # weights="distance"

# y_pred = cross_val_predict(clf, train_input, train_output['result'], cv=5)
# print(y_pred)

# Evaluate model
utils.modelEvaluation(clf, train_input, train_output, balance_classes=True, model_name="Test Classifier", fig_name="test")

