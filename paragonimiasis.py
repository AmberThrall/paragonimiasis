from paragonimiasis import *

CLASSIFIER = 'Result_of_ELISA_test'
MODEL_PARAMS = {
    'kernel': 'linear',
}

def main():
    print("Loading in data...", end='')
    df = pd.read_csv('prepared_data.csv', low_memory=False)
    print("Done!")

    cols = [ 
        'Have_you_had_Expectoration', 
        'Have_you_had_Chest_pain', 
    ]

    report = Report(df.columns, CLASSIFIER)

    clf, X, y_true = build_svm(df, cols, CLASSIFIER, MODEL_PARAMS) 
    result = test_model(clf, X, y_true)
    print(result)
    report.record(cols, result)

    cols.append('Consumption_of_cray_fishes')
    clf, X, y_true = build_svm(df, cols, CLASSIFIER, MODEL_PARAMS) 
    result = test_model(clf, X, y_true)
    print(result)
    report.record(cols, result)

    print(report._table)
    report.as_dataframe().to_csv('report.csv')

if __name__ == '__main__':
    main()
