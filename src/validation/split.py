import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVR
from evaluation_dir import evaluation_metrics
from split_train_test import convert_label
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def split_folding_data(X, y, n_folds):
    sss = StratifiedShuffleSplit(n_splits=n_folds, random_state=0)  # random_state = 0 -- default setting
    for train_index, test_index in sss.split(X, y):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        return (X_train, y_train), (X_test, y_test)


def baseline_algorithm(train, test, algorihm):
    X_train, y_train = train
    X_test, y_test = test
    X_train, X_test = preprocessing.scale(X_train), preprocessing.scale(X_test)
    if algorihm == 'svr_rbf':
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
        y_pred = model.fit(X_train, y_train).predict(X_test)
    elif algorihm == 'svr_poly':
        model = SVR(kernel='poly', C=1e3, degree=2)
        y_pred = model.fit(X_train, y_train).predict(X_test)
    elif algorihm == 'lr':
        model = LogisticRegression()
        y_pred = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    elif algorihm == 'svm':
        model = svm.SVC(probability=True).fit(X_train, y_train)
        y_pred = model.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    else:
        print('You need to give the correct algorithm name')

    acc, prc, rc, f1, auc_ = evaluation_metrics(y_true=y_test, y_pred=y_pred)
    print('Accuracy: %f -- Precision: %f -- Recall: %f -- F1: %f -- AUC: %f' % (acc, prc, rc, f1, auc_))