#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import statsmodels.api as sm
import joblib

# Initialize your data
data = [
    ['tuneunl', 105, 105, 6, 0, 6, 0, 6, 1, 48.6],
    ['bebas55+', 100, 100, 18, 0, 18, 0, 18, 1, 61],
    ['bebas55', 80, 80, 18, 0, 18, 0, 18, 1, 55],
    ['fokus40+', 86, 86, 8, 0, 8, 0, 8, 1, 47.2],
    ['fokus40', 66, 66, 8, 0, 8, 0, 8, 1, 41.5],
    ['fokus33+', 64, 64, 4, 0, 4, 0, 4, 1, 34.2],
    ['fokus33', 54, 54, 4, 0, 4, 0, 4, 1, 30.5],
    ['hi15', 10, 10, 18, 0, 1, 0, 18, 0, 6.7],
    ['hi25', 25, 25, 18, 0, 1, 0, 18, 0, 15.4],
    ['hi35', 35, 35, 18, 0, 18, 0, 18, 0, 21.2],
    ['m10-cun', 170, 10, 18, 60, 1, 100, 18, 1, 26.9],
    ['m10-value', 150, 10, 18, 40, 18, 100, 18, 1, 27.6],
    ['m6-cun', 121, 6, 18, 15, 1, 100, 18, 1, 21],
    ['m6-value', 112, 6, 18, 6, 18, 100, 18, 1, 21],
    ['bestbasic28', 30, 30, 3, 0, 3, 0, 3, 1, 19.9]
]

df = pd.DataFrame(data, columns=['plan_name', 'total_fup', 'main_bucket_fup', 'main_bucket_speed', 'video_fup', 'video_speed', 'social_fup', 'social_speed', 'unl_ind', 'actual_adpu'])

# Define core variables and model variables
core_variables = ['total_fup', 'main_bucket_fup', 'main_bucket_speed', 'video_fup', 'video_speed', 'social_fup', 'social_speed', 'unl_ind']
variables = core_variables + ['very_low_fup_ind', 'low_fup_ind', 'high_fup_ind', 'separated_video_fup_ind', 'separated_social_fup_ind', 'slower_video_speed_ind']

# Add indicator columns to the original data
df['very_low_fup_ind'] = df['total_fup'].apply(lambda x: 1 if x <= 10 else 0)
df['low_fup_ind'] = df['total_fup'].apply(lambda x: 1 if x <= 30 else 0)
df['high_fup_ind'] = df['total_fup'].apply(lambda x: 1 if x >= 80 else 0)
df['separated_video_fup_ind'] = df['video_fup'].apply(lambda x: 1 if x != 0 else 0)
df['separated_social_fup_ind'] = df['social_fup'].apply(lambda x: 1 if x != 0 else 0)
df['slower_video_speed_ind'] = df.apply(lambda row: 1 if row['video_speed'] != 0 and row['video_speed'] < row['main_bucket_speed'] else 0, axis=1)

# Prepare data for modeling
df_exog = sm.add_constant(df[variables])
df_endog = df['actual_adpu']

# Fit the model
poisson_model = sm.GLM(df_endog, df_exog, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

# Save the trained model to a file
joblib.dump(poisson_results, 'poisson_model.pkl')

# New plan data
new_plan = [
    [145, 30, 18, 50, 6, 50, 6, 1],  # all35
    [145, 30, 18, 50, 6, 50, 6, 0],  # all35-non-unl
    [35, 35, 18, 0, 18, 0, 18, 0],   # hi35
    [50, 50, 18, 0, 18, 0, 18, 0],   # 5g45
    [50, 50, 25, 0, 25, 0, 25, 0],   # 5g45v2
    [50, 50, 30, 0, 30, 0, 30, 0],   # 5g45v3
    [130, 130, 20, 0, 130, 0, 20, 0],  # 5g45v4
    [130, 130, 22, 0, 130, 0, 22, 0],  # 5g45v4
    [130, 130, 25, 0, 130, 0, 25, 0],  # 5g45v4
    [130, 130, 30, 0, 130, 0, 30, 0],  # 5g45v4
    [116, 15, 18, 0, 18, 100, 3, 1],   # start20
    [60, 10, 18, 25, 3, 25, 3, 1],     # tapau20v1
    [40, 10, 18, 15, 3, 15, 3, 1],     # tapau20v3
    [35, 5, 18, 15, 3, 15, 3, 1],      # tapau20v4
    [56, 6, 18, 0, 18, 50, 3, 1],      # BBNUv1
    [36, 6, 18, 0, 18, 30, 3, 1]       # BBNUv2
]

df_new_plan = pd.DataFrame(new_plan, columns=core_variables)

# Concatenate original data with new plan data
df_possible_combinations = pd.concat([df, df_new_plan], ignore_index=True)

# Add indicator columns to the new combined DataFrame
# Repeating the same steps as for the original df
df_possible_combinations['very_low_fup_ind'] = df_possible_combinations['total_fup'].apply(lambda x: 1 if x <= 10 else 0)
df_possible_combinations['low_fup_ind'] = df_possible_combinations['total_fup'].apply(lambda x: 1 if x <= 30 else 0)
df_possible_combinations['high_fup_ind'] = df_possible_combinations['total_fup'].apply(lambda x: 1 if x >= 80 else 0)
df_possible_combinations['separated_video_fup_ind'] = df_possible_combinations['video_fup'].apply(lambda x: 1 if x != 0 else 0)
df_possible_combinations['separated_social_fup_ind'] = df_possible_combinations['social_fup'].apply(lambda x: 1 if x != 0 else 0)
df_possible_combinations['slower_video_speed_ind'] = df_possible_combinations.apply(lambda row: 1 if row['video_speed'] != 0 and row['video_speed'] < row['main_bucket_speed'] else 0, axis=1)

# Ensure the DataFrame is in the correct format for prediction
df_possible_combinations = sm.add_constant(df_possible_combinations[variables])

# Predict ADPU for the possible combinations
df_possible_combinations['predicted_adpu'] = poisson_results.predict(df_possible_combinations[poisson_results.model.exog_names])

# Output the DataFrame with the predicted ADPU
print(df_possible_combinations[['plan_name', 'predicted_adpu']])
