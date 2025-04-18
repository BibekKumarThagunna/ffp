import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_PATH = 'ffp_model.joblib'
DATA_PATH = 'Clean_Dataset.csv'

st.set_page_config(page_title="Flight Fare Predictor", layout="wide")
st.title("✈️ Flight Fare Prediction")
st.write("Enter the details below to predict the flight fare.")

def load_model(path):
    if not os.path.exists(path):
        st.error(f"Error: Model file not found at {path}")
        st.stop()
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

@st.cache_data 
def load_data(path):
    if not os.path.exists(path):
        st.error(f"Error: Dataset file not found at {path}")
        st.stop()
    try:
        df = pd.read_csv(path)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading dataset for options: {e}")
        st.stop()

model_pipeline = load_model(MODEL_PATH)
df_options = load_data(DATA_PATH)

numerical_cols = ['duration', 'days_left']
categorical_cols = ['airline', 'source_city', 'destination_city', 'stops', 'class', 'departure_time', 'arrival_time']
expected_columns = numerical_cols + categorical_cols

csv_col_airline = 'airline'
csv_col_source = 'source_city'
csv_col_destination = 'destination_city'
csv_col_stops = 'stops'
csv_col_class = 'class'
csv_col_departure_time = 'departure_time'
csv_col_arrival_time = 'arrival_time'

col1, col2 = st.columns(2)

with col1:
    st.subheader("Flight Details")
    if csv_col_airline in df_options.columns:
        airline_selection = st.selectbox("Airline", sorted(df_options[csv_col_airline].unique()))
    else:
        st.error(f"Column '{csv_col_airline}' not found in data file.")
        st.stop()

    if csv_col_source in df_options.columns:
        source_selection = st.selectbox("Source City", sorted(df_options[csv_col_source].unique()))
    else:
        st.error(f"Column '{csv_col_source}' not found in data file. Please check column names in Clean_Dataset.csv.")
        st.stop()

    if csv_col_destination in df_options.columns:
        destination_selection = st.selectbox("Destination City", sorted(df_options[csv_col_destination].unique()))
    else:
        st.error(f"Column '{csv_col_destination}' not found in data file.")
        st.stop()

    if csv_col_stops in df_options.columns:
        stops_selection = st.selectbox("Number of Stops", sorted(df_options[csv_col_stops].unique()))
    else:
        st.error(f"Column '{csv_col_stops}' not found in data file.")
        st.stop()

    if csv_col_class in df_options.columns:
        class_selection = st.selectbox("Class", sorted(df_options[csv_col_class].unique()))
    else:
        st.error(f"Column '{csv_col_class}' not found in data file.")
        st.stop()

with col2:
    st.subheader("Timing and Duration")
    if csv_col_departure_time in df_options.columns:
        departure_time_selection = st.selectbox("Departure Time Category", sorted(df_options[csv_col_departure_time].unique()))
    else:
        st.error(f"Column '{csv_col_departure_time}' not found in data file.")
        st.stop()

    if csv_col_arrival_time in df_options.columns:
        arrival_time_selection = st.selectbox("Arrival Time Category", sorted(df_options[csv_col_arrival_time].unique()))
    else:
        st.error(f"Column '{csv_col_arrival_time}' not found in data file.")
        st.stop()

    duration_input = st.number_input("Duration (hours)", min_value=0.5, max_value=50.0, value=2.0, step=0.5)
    days_left_input = st.slider("Days Left Until Departure", min_value=1, max_value=50, value=15)

if st.button("Predict Fare", type="primary"):
    input_data = {
        'airline': [airline_selection],
        'source_city': [source_selection],
        'destination_city': [destination_selection],
        'stops': [stops_selection],
        'class': [class_selection],
        'departure_time': [departure_time_selection],
        'arrival_time': [arrival_time_selection],
        'duration': [duration_input],
        'days_left': [days_left_input]
    }

    input_df = pd.DataFrame(input_data)

    try:
        input_df = input_df[expected_columns]
    except KeyError as e:
        st.error(f"Error: Mismatch between expected columns for model and created DataFrame: {e}. Ensure 'categorical_cols' list matches training. Columns provided: {list(input_df.columns)}")
        st.stop()
    except Exception as e:
         st.error(f"Error during column reordering: {e}")
         st.stop()

    try:
        prediction = model_pipeline.predict(input_df)
        predicted_price = prediction[0]
        st.success(f"Predicted Flight Fare: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Input Data Sent to Model (after column reordering):")
        st.dataframe(input_df)
