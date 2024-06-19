import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from .report import *

def prepare_data(df, columns, classifier):
    # Build subset of dataframe
    subset = columns.copy()
    subset.append(classifier)
    df_subset = df.copy()[subset]

    # Perform One-Hot encoding on categorical data
    df_hot_encoded = pd.get_dummies(df_subset)
    
    # Construct SVM data table and classifier array
    X = df_hot_encoded[df_hot_encoded.columns[df_hot_encoded.columns!=classifier]].values
    y = df_hot_encoded[classifier].values

    return X, y

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

def learn(df, classifier, model_params):
    feature_names = []
    for column in df.columns:
        if column != classifier and column != 'Unnamed: 0':
            feature_names.append(column)

    X, y = prepare_data(df, feature_names, classifier)
    feature_names = np.array(feature_names)

    # Split the data set into training and testing
#       X_train, X_test, y_train, y_test = train_test_split(X, y, 
#        test_size=0.3, random_state=0)

    # Perform Sequence feature selection
    scorer = make_scorer(matthews_corrcoef)
    clf_svm = svm.SVC(**model_params)
    clf_sfs = SequentialFeatureSelector(
        clf_svm,
        scoring = scorer,
        n_jobs = -1 # Use all processors
    )
    clf_sfs.fit(X, y)

    good_features = list(feature_names[clf_sfs.get_support()])

    # Test the new model
    X, y = prepare_data(df, good_features, classifier)
    clf_svm = svm.SVC(**model_params).fit(X, y)
    report = test_model(clf_svm, X, y)
    report['selected_features'] = good_features
    return clf_svm, report

