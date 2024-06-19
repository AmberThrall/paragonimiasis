import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, matthews_corrcoef, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.pipeline import make_pipeline
from .report import *

KFOLD_N_SPLITS = 5

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

def learn(df, classifier, param_grid):
    feature_names = []
    for column in df.columns:
        if column != classifier and column != 'Unnamed: 0':
            feature_names.append(column)

    X, y = prepare_data(df, feature_names, classifier)
    feature_names = np.array(feature_names)

    # Split the data set into training and testing
    print(" - Splitting the data into training and testing set...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
        test_size=0.33, random_state=42)

    # Perform grid-search
    print(" - Performing grid search to find optimal parameters...")
    scorer = make_scorer(matthews_corrcoef)
    clf_svm = svm.SVC()
    clf_gs = GridSearchCV(
        estimator=clf_svm, 
        param_grid=param_grid, 
        scoring=scorer, 
        refit=True,
        cv=KFOLD_N_SPLITS
    )
    clf_gs.fit(X_train, y_train)

    # Rebuild SVM with best parameters
    best_params = clf_gs.best_estimator_.get_params()
    clf_best_svm = clf_svm.set_params(**best_params)

    # Perform Sequence feature selection
    print(" - Performing sequential feature selection to find optimal features...")
    clf_sfs = SequentialFeatureSelector(
        clf_best_svm,
        scoring = scorer,
        cv = KFOLD_N_SPLITS,
        #n_jobs = -1 # Use all processors
    )
    clf_sfs.fit(X_train, y_train)

    good_features = list(feature_names[clf_sfs.get_support()])

    # Test the new model
    print(" - Testing model on test set...")
    X_train = X_train[:,clf_sfs.get_support()]
    X_test = X_test[:,clf_sfs.get_support()]
    clf_svm = svm.SVC(**best_params).fit(X_train, y_train)
    report = test_model(clf_svm, X_test, y_test)
    report['selected_features'] = good_features
    report['best_params'] = best_params
    return clf_svm, report

