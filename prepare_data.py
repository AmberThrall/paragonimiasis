import pandas as pd
import numpy as np
import re

SYMPTOMS = {
    'Have_you_had_Cough': 'Duration_of_cough',
    'Have_you_had_Expectoration': 'Duration_of_expectoration',
    'Have_you_had_Chest_pain': 'Duration_of_chest_pain',
    'Have_you_lost_appetite': 'Duration_of_loss_of_appetite',
    'Have_you_had_blood_in_sputum': 'Duration_of_blood_in_sputum',
    'Have_you_had_night_sweats': 'Duration_of_night_sweats',
    'Have_you_lost_weight': 'Duration_of_weight_loss',
    'Have_you_had_shortness_of_breath': 'Duration_of_shortness_of_breath',
    'Have_you_had_tiredness': 'Duration_of_tiredness',
}

CONSUMPTION_METHODS = {
    'Consumption_of_fresh_water_crabs': {
        'dur': 'Duration_of_consumption_of_fresh_water_crabs',
        'freq': 'Frequency_of_consumption_of_fresh_water_crabs',
    },
    'Consumption_of_cray_fishes': {
        'dur': 'Duration_of_consumption_of_cray_fishes',
        'freq': 'Frequency_of_consumption_of_cray_fishes',
    },
    'Consumption_of_wlid_boar_meat': {
        'dur': 'Duration_of_consumption_of_wild_boar_meat',
        'freq': 'Frequency_of_consumption_of_wild_boar_meat',
    },
    'Consumption_of_Rodents_Rats_etc': {
        'dur': 'Duration_of_consumption_of_rodents',
        'freq': 'Frequency_of_consumption_of_rodents',
    }
}

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
    'Other_symptoms',
    'Have_you_taken_any_treatment_or_actions',
    'Have_you_ever_been_treated_for_TB',
    'Raw_fresh_water_crabs',
    'Roasted_fresh_water_crabs',
    'Smoked_fresh_water_crabs',
    'Soup_fresh_water_crabs',
    'Pickled_fresh_water_crabs',
    'Cooked_fresh_water_crabs',
    'Raw_cray_fish',
    'Roasted_cray_fish',
    'Smoked_cray_fish',
    'Soup_cray_fish',
    'Pickled_cray_fish',
    'Cooked_cray_fish',
    'Raw_wild_boar_meat',
    'Roasted_wild_boar_meat',
    'Smoked_wild_boar_meat',
    'Soup_wild_boar_meat',
    'Pickled_wild_boar_meat',
    'Cooked_wild_boar_meat',
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
    #one_values = ['Positive']
    #zero_values = ['Negative']
    if x in one_values:
        return 1
    elif x in zero_values:
        return 0
    else:
        return x

def convert_dur_str(x):
    if not type(x) == str:
        return x
        
    x = x.strip().lower()
    try:
        split = re.split(r'(\d+)', x)
        dur = int(split[1])
        if split[2].strip() == 'weeks' or split[2].strip() == 'week':
            dur *= 7
        if split[2].strip() == 'months' or split[2].strip() == 'month':
            dur *= 30
        return dur
    except:
        return np.nan

def main():
    for key, value in SYMPTOMS.items():
        DESIRED_COLUMNS.append(key)
        DESIRED_COLUMNS.append(value)

    for key, value in CONSUMPTION_METHODS.items():
        DESIRED_COLUMNS.append(key)
        DESIRED_COLUMNS.append(value['dur'])
        DESIRED_COLUMNS.append(value['freq'])

    df = pd.read_csv('data.csv', low_memory=False)
    df = df[DESIRED_COLUMNS].copy()
    transform_all_columns(df, strip_whitespace)
    transform_all_columns(df, convert_binary)

    ############
    # SYMPTOMS #
    ############
    # Fill NaN where the symptom is not present with zeroes
    for key, value in SYMPTOMS.items():
        patients_without_symptom = df[df[key] == 0]
        df.loc[df[key] == 0, value] = patients_without_symptom[value].apply(lambda x: 0)

    # Convert duration strings
    for key, value in SYMPTOMS.items():
        df[value] = df[value].apply(convert_dur_str)

    # Drop yes/no symptom columns as they are now covered by duration
    df = df.drop(columns=SYMPTOMS.keys())

    #######################
    # CONSUMPTION METHODS #
    #######################
    # Fill NaN for freq/dur where consumption method is no with N/A
    for key, value in CONSUMPTION_METHODS.items():
        patients_without_method = df[df[key] == 0]
        df.loc[df[key] == 0, value['dur']] = patients_without_method[value['dur']].apply(lambda x: 'N/A')
        df.loc[df[key] == 0, value['freq']] = patients_without_method[value['freq']].apply(lambda x: 'N/A')
    
    # Drop yes/no conumption columns as they are now covered by freq/dur
    df = df.drop(columns=CONSUMPTION_METHODS.keys())

    #####################
    # DROP ROWS/COLUMNS #
    #####################
    # Drop columns with no differing values
    bad_columns = []
    for column in df.columns:
        if len(df[column].value_counts().to_dict().keys()) <= 1:
            bad_columns.append(column)

    df = df.drop(columns = bad_columns)

    # Drop rows with NA entries
    df = df.dropna(how='any').copy()

    ##############
    # HOT ENCODE #
    ##############
    df = pd.get_dummies(df)

    # Save the result
    df.to_csv('prepared_data.csv')

if __name__ == '__main__':
    main()
