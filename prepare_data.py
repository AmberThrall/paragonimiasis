import pandas as pd
import numpy as np

DESIRED_COLUMNS = [
    'Age_of_the_study_participant',
    'Sex_of_the_study_participant',
    'Marital_status_of_the_participant',
    'Religion_of_the_participant',
    'Belongs_to_tribal_community',
    'Height_of_the_study_participant_in_Cms',
    'Weight_of_the_study_participant_in_Kgs',
    'Educational_qualification_of_the_study_participant',
    'Occupation_of_the_study_participant',
    'Which_health_facility_do_you_access_when_you_are_sick',
    'Have_you_had_Cough',
    'Have_you_had_Expectoration',
    'Have_you_had_Chest_pain',
    'Have_you_had_fever',
    'Have_you_lost_appetite',
    'Have_you_had_blood_in_sputum',
    'Have_you_had_night_sweats',
    'Have_you_lost_weight',
    'Have_you_had_shortness_of_breath',
    'Have_you_had_tiredness',
    'Other_symptoms',
    'Have_you_taken_any_treatment_or_actions',
    'Have_you_ever_been_treated_for_TB',
    'Consumption_of_fresh_water_crabs',
    'Raw_fresh_water_crabs',
    'Roasted_fresh_water_crabs',
    'Smoked_fresh_water_crabs',
    'Soup_fresh_water_crabs',
    'Pickled_fresh_water_crabs',
    'Cooked_fresh_water_crabs',
    'Consumption_of_cray_fishes',
    'Raw_cray_fish',
    'Roasted_cray_fish',
    'Smoked_cray_fish',
    'Soup_cray_fish',
    'Pickled_cray_fish',
    'Cooked_cray_fish',
    'Consumption_of_wlid_boar_meat',
    'Raw_wild_boar_meat',
    'Roasted_wild_boar_meat',
    'Smoked_wild_boar_meat',
    'Soup_wild_boar_meat',
    'Pickled_wild_boar_meat',
    'Cooked_wild_boar_meat',
    'Consumption_of_Rodents_Rats_etc',
    'Raw_rodents',
    'Roasted_rodents',
    'Smoked_rodents',
    'Soup_rodents',
    'Pickled_rodents',
    'Cooked_rodents',
    'Result_of_ELISA_test'
]

def transform_all_columns(df, f):
    for column in df.columns:
        df[column] = df[column].apply(f)

def strip_whitespace(x):
    if type(x) == str:
        return x.strip()
    return x
    
def convert_binary(x):
    one_values = ['Yes', 'Positive', 'Married', 'Yes Currently on ATT']
    zero_values = ['No', 'Negative', 'Unmarried']
    if x in one_values:
        return 1
    elif x in zero_values:
        return 0
    else:
        return x

def main():
    df = pd.read_csv('data.csv', low_memory=False)
    df = df[DESIRED_COLUMNS].copy()
    transform_all_columns(df, strip_whitespace)
    transform_all_columns(df, convert_binary)

    # Drop columns with no differing values
    bad_columns = []
    for column in df.columns:
        if len(df[column].value_counts().to_dict().keys()) <= 1:
            bad_columns.append(column)

    df = df.drop(columns = bad_columns)

    # Drop rows with NA entries
    df = df.dropna(how='any').copy()

    # Save the result
    df.to_csv('prepared_data.csv')

if __name__ == '__main__':
    main()
