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

#Early Fusion
y = np.load('./data/trab2_dataset.npz')['labels']
X1 = load_data('./data/trab2_conv1.npz', type_l='conv')
X2 = load_data('./data/trab2_conv5.npz', type_l='conv')
X3 = load_data('./data/trab2_dense2.npz', type_l='dense')
X = np.concatenate( (X1,X2,X3), axis=1 )
#Shuffle, divide and normalize data
X, y_shuffle = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y_shuffle, test_size=0.33, random_state=0)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {'gamma': [1e-3, 1e-4, 1e-5], 'C': [1, 10, 100, 1000]}
svm_classifier = SVC(cache_size=1024, kernel='rbf')
grid = GridSearchCV(svm_classifier, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

#Execucao para o melhor fit
y_pred = grid.predict(X_test)

#Matriz de confusão - heatmap
mean = grid.cv_results_['mean_test_score'][grid.best_index_]
std = grid.cv_results_['std_test_score'][grid.best_index_]
cnf_matrix = confusion_matrix(y_test, y_pred)
print "\n\n--- Results for early fusion and {0} --- \n".format(grid.best_params_)
print("Overall accuracy: %0.3f +/-%0.03f" % (mean, std * 2))
print("Average accuracy: %0.3f +/-%0.03f" % (AverageAccuracy(cnf_matrix)))
plot_confusion_matrix(cnf_matrix, classes=labels_name, normalize=False)
plt.show()
