from paragonimiasis import *
import pprint

CLASSIFIER = 'Result_of_ELISA_test'
TEST_PARAMS = {
    'linear': [{ 'C': np.linspace(1, 25, num=6), 'kernel': ['linear'] }],
    'rbf': [{ 
        'C': np.linspace(1, 25, num=6), 
        'kernel': ['rbf'],  
        'gamma': np.linspace(0.001, 0.01, num=10),
    }],
}

def print_report(report):
    print("\n-------")
    print("Report:")
    print("-------")
    print("Accuracy: {}".format(report['accuracy']))
    print("MCC: {}".format(report['mcc']))
    print("Confusion Matrix:")
    print(report['confusion_matrix'])
    print("Selected Parameters:")
    params = pd.Series(report['best_params'])
    print(params)
    print("Selected Parameters:")
    for feat in report['selected_features']:
        print(" - {}".format(str(feat)))

def main():
    print("Loading in data...", end='')
    df = pd.read_csv('prepared_data.csv', low_memory=False)
    #df = df[['Result_of_ELISA_test', 'Age_of_the_study_participant', 'Height_of_the_study_participant_in_Cms']].copy()
    print("Done!")

    for key, params in TEST_PARAMS.items():
        print("\n\nRunning test '{}'...".format(key))
        model, report = learn(df, CLASSIFIER, params)
        print_report(report)
    
if __name__ == '__main__':
    main()
