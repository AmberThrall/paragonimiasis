from paragonimiasis import *

CLASSIFIER = 'Result_of_ELISA_test'
MODEL_PARAMS = {
    'kernel': 'linear',
}

def main():
    print("Loading in data...", end='')
    df = pd.read_csv('prepared_data.csv', low_memory=False)
    #df = df[['Result_of_ELISA_test', 'Age_of_the_study_participant', 'Height_of_the_study_participant_in_Cms']].copy()
    print("Done!")

    print("Training...")
    model, report = learn(df, CLASSIFIER, MODEL_PARAMS)

    print("\nReport:")
    print("-------")
    print("Accuracy: {}".format(report['accuracy']))
    print("MCC: {}".format(report['mcc']))
    print("Confusion Matrix:")
    print(report['confusion_matrix'])
    print("Selected Features: {}".format([str(x) for x in report['selected_features']]))


if __name__ == '__main__':
    main()
