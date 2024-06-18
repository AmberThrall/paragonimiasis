import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, matthews_corrcoef

def build_svm(df, columns, classifier, model_params={}):
    # Build subset of dataframe
    subset = columns.copy()
    subset.append(classifier)
    df_subset = df.copy()[subset]

    # Perform One-Hot encoding on categorical data
    df_hot_encoded = pd.get_dummies(df_subset)
    
    # Construct SVM data table and classifier array
    X = df_hot_encoded[df_hot_encoded.columns[df_hot_encoded.columns!=classifier]].values
    y_true = df_hot_encoded[classifier].values

    clf = svm.SVC(**model_params).fit(X, y_true)
    return clf, X, y_true

def mcc(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    return matthews_corrcoef(y_test, y_pred)


def build_confusion_matrix(clf, X_test, y_test, normalize='true'):
    y_pred = clf.predict(X_test)
    return confusion_matrix(y_test, y_pred, labels=clf.classes_, normalize=normalize)

def test_model(clf, X_test, y_test):
    return {
        'accuracy': clf.score(X_test, y_test),
        'mcc': mcc(clf, X_test, y_test),
        'confusion_matrix': build_confusion_matrix(clf, X_test, y_test),
    }
