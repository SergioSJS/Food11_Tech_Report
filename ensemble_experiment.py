#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np, itertools, math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle, resample
from utils import *

X = load_data('./data/trab2_dense2.npz', 'dense')
y = np.load('./data/trab2_dataset.npz')['labels']
X, y_shuffle = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y_shuffle, test_size=0.33, random_state=0)

#Normalize data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#4096 features + Random Forest + GridSearch
param_grid_rf = {"max_depth": [3, None], "max_features": [1, 3, 10], "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10], "bootstrap": [True, False], "criterion": ["gini", "entropy"]}
clf_rf = RandomForestClassifier(n_estimators=20)
grid = GridSearchCV(clf_rf, param_grid_rf, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
grid_mean = grid.cv_results_['mean_test_score'][grid.best_index_]
grid_std = grid.cv_results_['std_test_score'][grid.best_index_]
y_pred = grid.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)

print("--- Best results (Random Forest) found with %r ---" % (grid.best_params_))
print("Overall accuracy: %0.3f +/-%0.03f" % ( grid_mean, grid_std * 2 ))
print("Average accuracy: %0.3f +/-%0.03f" % (AverageAccuracy(cnf_matrix)))
plot_confusion_matrix(cnf_matrix, classes=labels_name, normalize=False)
plt.show()

#5-subset features + Linear SVM + Majority Voting
n_subset = 5
y_pred = []
grid_accuracy = []
clf_svm = SVC(kernel='linear')
param_grid = {'gamma': [1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
slice_size = int(math.ceil(4096/n_subset))
for n in xrange(n_subset):
    start_idx = n*n_subset
    end_idx = n*n_subset+slice_size
    X_train_subset = X_train[:,start_idx:end_idx]
    X_test_subset = X_test[:,start_idx:end_idx]
    grid = GridSearchCV(clf_svm, param_grid, cv=5, n_jobs=-1)
    grid.fit(X_train_subset, y_train)
    y_pred.append( grid.predict(X_test_subset) )
    grid_accuracy.append( (grid.cv_results_['mean_test_score'][grid.best_index_],grid.cv_results_['std_test_score'][grid.best_index_]) )
mean_accuracy = np.mean( [x[0] for x in grid_accuracy] )
mean_std = np.mean( [x[1] for x in grid_accuracy] )
print("--- Best results (Majority Voting) ---")
y_pred = MajorityVote(y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Overall accuracy: %0.3f +/-%0.03f" % (mean_accuracy, mean_std * 2))
print("Average accuracy: %0.3f +/-%0.03f" % (AverageAccuracy(cnf_matrix)))
plot_confusion_matrix(cnf_matrix, classes=labels_name, normalize=False)
plt.show()

#5-subset features + Bagging (11 model)
n_subset = 5
bag_size = 11
scores = []
slice_size = int(math.ceil(4096/n_subset))
y_pred = []
subset_accuracy = []
clf_svm = SVC(kernel='linear')
for n in xrange(n_subset):
    start_idx = n*n_subset
    end_idx = n*n_subset+slice_size
    X_train_subset = X_train[:,start_idx:end_idx]
    X_test_subset = X_test[:,start_idx:end_idx]
    y_subset_pred = []
    grid_subset_accuracy = []
    for b in xrange(bag_size):
        X_bagg, y_bagg = resample(X_train_subset, y_train)
        grid = GridSearchCV(clf_svm, param_grid, cv=5, n_jobs=-1)
        grid.fit(X_bagg, y_bagg)
        y_subset_pred.append( grid.predict(X_test_subset) )
        grid_subset_accuracy.append( grid.cv_results_['mean_test_score'][grid.best_index_] )
    subset_accuracy.append( np.mean(grid_subset_accuracy) )
    y_pred.append( MajorityVote(y_subset_pred) )

mean_accuracy = np.mean( subset_accuracy )
mean_std = np.std( subset_accuracy )
print("--- Best results (Bagging) --- ")
#Confusion matrix
y_pred = MajorityVote(y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Overall accuracy: %0.3f +/-%0.03f" % (mean_accuracy, mean_std * 2))
print("Average accuracy: %0.3f +/-%0.03f" % (AverageAccuracy(cnf_matrix)))
plot_confusion_matrix(cnf_matrix, classes=labels_name, normalize=False)
plt.show()
