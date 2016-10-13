import _pickle as cPickle
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from sklearn import neighbors
from sklearn.svm import SVC
#from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import operator
import re
from dateutil import parser


def performRFClass(X_train, y_train, X_test, y_test, parameters=None, fout=None, savemodel=False):
    """
    Random Forest Binary Classification
    """
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf.fit(X_train, y_train)
        
    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy



def performKNNClass(X_train, y_train, X_test, y_test, parameters=None, fout=None, savemodel=False):
    """
    KNN binary Classification
    """
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy


def performSVMClass(X_train, y_train, X_test, y_test, parameters=None, fout=None, savemodel=False):
    """
    SVM binary Classification
    """
    #c = parameters[0]
    #g =  parameters[1]
    clf = SVC()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy


def performAdaBoostClass(X_train, y_train, X_test, y_test, parameters=None, fout=None, savemodel=False):
    """
    Ada Boosting binary Classification
    """
    #n = parameters[0]
    #l =  parameters[1]
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy


def performGTBClass(X_train, y_train, X_test, y_test, parameters=None, fout=None, savemodel=False):
    """
    Gradient Tree Boosting binary Classification
    """
    clf = GradientBoostingClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy

def performQDAClass(X_train, y_train, X_test, y_test, parameters=None, fout=None, savemodel=False):
    """
    Quadratic Discriminant Analysis binary Classification
    """
    def replaceTiny(x):
        if (abs(x) < 0.0001):
            x = 0.0001

    X_train = X_train.apply(replaceTiny)
    X_test = X_test.apply(replaceTiny)
    
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)

    if savemodel == True:
        fname_out = '{}-{}.pickle'.format(fout, datetime.now())
        with open(fname_out, 'wb') as f:
            cPickle.dump(clf, f, -1)    
    
    accuracy = clf.score(X_test, y_test)
    
    return accuracy


def performMLPClass(X_train, y_train, X_test, y_test, parameters=None, fout=None, savemodel=False):
    """
    Multi-layer Perceptron neural network Classification
    """
    layers = (5,6,3)
    if parameters is not None:
        layers = parameters

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=layers, random_state=1)
    mlp.fit(X_train, y_train) 
    accuracy = mlp.score(X_test, y_test)
    return accuracy, mlp

