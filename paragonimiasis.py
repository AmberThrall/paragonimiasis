from paragonimiasis import *

CLASSIFIER = 'Result_of_ELISA_test'
MODEL_PARAMS = {
    'kernel': 'linear',
}

def main():
    print("Loading in data...", end='')
    df = pd.read_csv('prepared_data.csv', low_memory=False)
    print("Done!")

    print("Training...")
    report = learn(df, CLASSIFIER, MODEL_PARAMS)
    report.as_dataframe().to_csv('report.csv')

if __name__ == '__main__':
    main()
