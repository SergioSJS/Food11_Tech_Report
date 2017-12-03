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
from sklearn.model_selection import KFold
from sklearn.utils import shuffle, resample
from utils import *

#Late Fusion - Boosting
#kf = KFold(n_splits=5)
y = np.load('./data/trab2_dataset.npz')['labels']
X1 = load_data('./data/trab2_conv1.npz', type_l='conv')
X2 = load_data('./data/trab2_conv5.npz', type_l='conv')
X3 = load_data('./data/trab2_dense2.npz', type_l='dense')

#Shuffle and divide data
X1, X2, X3, y_shuffle = shuffle(X1, X2, X3, y, random_state=0)
indices = np.random.permutation(X1.shape[0])
l1 = int(len(indices)*.7)
training_idx, test_idx = indices[:l1], indices[l1:]
y_train, y_test = y_shuffle[training_idx], y_shuffle[test_idx]

#Scalling data - normalization
X1_train, X1_test = X1[training_idx], X1[test_idx]
scaler = StandardScaler()
scaler.fit(X1_train)
X1_train = scaler.transform(X1_train)
X1_test = scaler.transform(X1_test)

X2_train, X2_test = X2[training_idx], X2[test_idx]
scaler = StandardScaler()
scaler.fit(X2_train)
X2_train = scaler.transform(X2_train)
X2_test = scaler.transform(X2_test)

X3_train, X3_test = X3[training_idx], X3[test_idx]
scaler = StandardScaler()
scaler.fit(X3_train)
X3_train = scaler.transform(X3_train)
X3_test = scaler.transform(X3_test)


param_grid = {'gamma': [1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
#Grid search e treinamento - cv1
clf1 = SVC(cache_size=512, kernel='linear')
grid1 = GridSearchCV(clf1, param_grid, cv=5, n_jobs=-1)
grid1.fit(X1_train, y_train)
y1_pred = grid1.predict(X1_test)

#Grid search e treinamento - cv5
clf2 = SVC(cache_size=512, kernel='linear')
grid2 = GridSearchCV(clf2, param_grid, cv=5, n_jobs=-1)
grid2.fit(X2_train, y_train)
y2_pred = grid2.predict(X2_test)

#Grid search e treinamento - dense2
clf3 = SVC(cache_size=512, kernel='linear')
grid3 = GridSearchCV(clf3, param_grid, cv=5, n_jobs=-1)
grid3.fit(X3_train, y_train)
y3_pred = grid3.predict(X3_test)

#print "\n\n--- Results for late fusion ---\n"
mean1 = grid1.cv_results_['mean_test_score'][grid1.best_index_]
std = grid1.cv_results_['std_test_score'][grid1.best_index_]
#print("Best results (grid1) found with %r. Mean and std_dev: %0.3f (+/-%0.03f)" % (grid1.best_params_, mean1, std * 2))
mean2 = grid2.cv_results_['mean_test_score'][grid2.best_index_]
std = grid2.cv_results_['std_test_score'][grid2.best_index_]
#print("Best results (grid2) found with %r. Mean and std_dev: %0.3f (+/-%0.03f)" % (grid2.best_params_, mean2, std * 2))
mean3 = grid3.cv_results_['mean_test_score'][grid3.best_index_]
std = grid3.cv_results_['std_test_score'][grid3.best_index_]
#print("Best results (grid3) found with %r. Mean and std_dev: %0.3f (+/-%0.03f)" % (grid3.best_params_, mean3, std * 2))
y_pred = MajorityVote([y1_pred, y2_pred, y3_pred])
cnf_matrix = confusion_matrix(y_test, y_pred)

print "\n\n--- Results for late fusion --- \n"
mm = np.array( [mean1, mean2, mean3] )
print("Overall accuracy: %0.3f +/-%0.03f" % ( np.mean(mm), np.std(mm) ))
print("Average accuracy: %0.3f +/-%0.03f" % (AverageAccuracy(cnf_matrix)))
plot_confusion_matrix(cnf_matrix, classes=labels_name, normalize=False)
plt.show()
