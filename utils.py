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

np.set_printoptions(precision=2)
plt.rcParams["figure.figsize"] = (7,7)
labels_name = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']

#Plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.gnuplot):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_aspect('auto')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    
def load_data(layer_path, type_l='conv'):
    layer = np.load(layer_path)['data']
    if type_l == 'conv':
        n_img, n_feat, sz_conv, _ = layer.shape
        X = layer.reshape( (n_img,n_feat*sz_conv*sz_conv) )
    else:
        X = layer
    return X

'''
Perform hard majority vote
'''
def MajorityVote(predictions):
    pred = np.asarray([predictions]).T
    b = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred.astype('int'))
    #Finally, reverse transform the labels for correct output:
    return b.T[0].tolist()

def AverageAccuracy(a):
    aa = np.diag(a) / np.sum(a,axis=1).astype(float)
    return (np.mean(aa), np.std(aa))

