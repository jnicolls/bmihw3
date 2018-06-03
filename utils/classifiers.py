'''
	classifiers.py
'''

import numpy as np
import pandas as pd

from sklearn import preprocessing

from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from features import *
from file_ops import *


def train_models(train, labels):
    
    svcClf = SVC(random_state = 260)
    svcClf.fit(train, labels)
    
    rfClf = RandomForestClassifier(random_state = 260)
    rfClf.fit(train, labels)
    
    return [svcClf, rfClf]


def Tuner(combined_features, groundtruth):
    '''
    Input: 
        * all features
        * ground truth labels

    Output:
        * tuning grid of (accuracy, sensitivity, specificity) for specified bounds on C, class_weight

    '''
    C = .3
    C_bound = 20
    grid = np.zeros((C_bound, 3))
    for i in range(C_bound):
        grid[i] = EvaluateClassifier(combined_features, groundtruth, C)
        C += .3
    return grid


def EvaluateClassifier(X, Y, C):
    '''
    Input: 
        * Dataset X
        * Labels Y
    Output:
        * outputs performance metrics: accuracy, sens, spec

    Evaluates a linear-kernel support classifer with given penalty input C and melanoma class-weight weight
    with 10-fold cross-validation. We must adjust the class weight to account for the class imbalance inherent 
    to the dataset. 
    '''

    svm_man = SVC(C = C, kernel='linear')

    scores = cross_val_score(svm_man, X, Y, cv=10, scoring='accuracy')
    acc = scores.mean()
    print('acc is:')
    print(acc)

    #------ Compute metrics ---------
    y_predict = cross_val_predict(svm_man, X, Y, cv=10)
    conf_mat = confusion_matrix(Y, y_predict)
    tn, fp, fn, tp = conf_mat.ravel()
    sens = float(tp)/(tp+fn)
    spec = float(tn)/(tn+fp)
    print('sens is:')
    print(sens)
    print('spec is:')
    print(spec)
    print(conf_mat)

    return (acc, sens, spec)


def validate_models(models, test, y_true):
    '''
        input: 
            * models, a list of trained models 
            * test, the test set on which models will be evaluated
            * feature_names, the list of feature columns that are being used 
    '''
    print("Preparing to validate models")
    for model in models:
        y_pred = model.predict(test)
        conf_mat = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_mat.ravel()
        sens = float(tp)/(tp+fn)
        spec = float(tn)/(tn+fp)
        print('sens is:')
        print(sens)
        print('spec is:')
        print(spec)
        print(conf_mat)

        print("auroc: " +  str(roc_auc_score(y_true, y_pred)))
        
    print("Done validating models")
    