#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
import joblib
import statsmodels.api as sm  # Ensure statsmodels is imported

# Load the pre-trained Poisson model
poisson_model = joblib.load('poisson_model.pkl')

# Streamlit app title
st.title("ADPU Prediction App")

# Function to create indicator variables based on the input
def create_indicators(input_data):
    input_data['very_low_fup_ind'] = int(input_data['total_fup'] <= 10)
    input_data['low_fup_ind'] = int(input_data['total_fup'] <= 30)
    input_data['high_fup_ind'] = int(input_data['total_fup'] >= 80)
    input_data['separated_video_fup_ind'] = int(input_data['video_fup'] > 0)
    input_data['separated_social_fup_ind'] = int(input_data['social_fup'] > 0)
    input_data['slower_video_speed_ind'] = int(input_data['video_speed'] > 0 and input_data['video_speed'] < input_data['main_bucket_speed'])
    return input_data

# Session state for storing multiple plans
if 'plans' not in st.session_state:
    st.session_state['plans'] = []

# UI for entering custom plan parameters
with st.sidebar.form("plan_form", clear_on_submit=True):
    st.header("Enter Plan Details")
    new_plan_name = st.text_input("New Plan Name")
    total_fup = st.number_input("Total FUP", min_value=0.0, format="%.2f")
    main_bucket_fup = st.number_input("Main Bucket FUP", min_value=0.0, format="%.2f")
    main_bucket_speed = st.number_input("Main Bucket Speed", min_value=0.0, format="%.2f")
    video_fup = st.number_input("Video FUP", min_value=0.0, format="%.2f")
    video_speed = st.number_input("Video Speed", min_value=0.0, format="%.2f")
    social_fup = st.number_input("Social FUP", min_value=0.0, format="%.2f")
    social_speed = st.number_input("Social Speed", min_value=0.0, format="%.2f")
    unl_ind = st.selectbox("Unlimited Indicator", [0, 1])
    
    # Button to add plan to session state
    submit_button = st.form_submit_button(label="Add Plan")

if submit_button:
    input_data = {
        'plan_name': new_plan_name,
        'total_fup': total_fup,
        'main_bucket_fup': main_bucket_fup,
        'main_bucket_speed': main_bucket_speed,
        'video_fup': video_fup,
        'video_speed': video_speed,
        'social_fup': social_fup,
        'social_speed': social_speed,
        'unl_ind': unl_ind
    }
    input_data = create_indicators(input_data)
    st.session_state['plans'].append(input_data)
    st.experimental_rerun()

# Displaying the session state plans
if st.session_state['plans']:
    st.write("Plans to Predict:")
    plans_df = pd.DataFrame(st.session_state['plans'])
    st.write(plans_df)

    # Predicting ADPU for all plans in the session state
    if st.button("Predict ADPU for All Plans"):
        predictions = []
        for plan in st.session_state['plans']:
            plan_df = pd.DataFrame([plan])
            plan_df = sm.add_constant(plan_df, has_constant='add')  # Add a constant term for the intercept
            # Ensure only the features used in the model are in the DataFrame
            prediction_df = plan_df[poisson_model.model.exog_names]
            prediction = poisson_model.predict(prediction_df)
            predictions.append(prediction.iloc[0])
        
        plans_df['predicted_adpu'] = predictions
        st.write("Predicted ADPU for Plans:")
        st.write(plans_df)
