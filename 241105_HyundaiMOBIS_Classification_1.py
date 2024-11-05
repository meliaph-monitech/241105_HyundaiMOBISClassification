import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Function to load the trained model
def load_model(model_path):
    return joblib.load(model_path)

# Function to load and filter the new CSV file
def load_and_filter_csv(file):
    df = pd.read_csv(file)
    df_filtered = df[df['L/O'] >= 0.4]  # Apply the filtering based on L/O column
    return df_filtered[['NIR', 'VIS']]

# Function to extract features from the new data
def extract_features(df_filtered):
    return {
        'mean_NIR': np.mean(df_filtered['NIR']),
        'std_NIR': np.std(df_filtered['NIR']),
        'max_NIR': np.max(df_filtered['NIR']),
        'min_NIR': np.min(df_filtered['NIR']),
        'mean_VIS': np.mean(df_filtered['VIS']),
        'std_VIS': np.std(df_filtered['VIS']),
        'max_VIS': np.max(df_filtered['VIS']),
        'min_VIS': np.min(df_filtered['VIS']),
    }

# Streamlit UI
st.title("Laser Welding Signal Classification")

# Upload the trained model
model_file = st.file_uploader("Upload the trained model (joblib file)", type=["joblib"])
if model_file is not None:
    model = load_model(model_file)  # Load the trained model

# Upload new CSV file for classification
uploaded_file = st.file_uploader("Upload a CSV file for classification", type=["csv"])
if uploaded_file is not None:
    # Load and preprocess the new data
    df_filtered = load_and_filter_csv(uploaded_file)
    
    if not df_filtered.empty:
        # Extract features for classification
        features = extract_features(df_filtered)
        feature_array = np.array(list(features.values())).reshape(1, -1)  # Reshape for model input
        
        # Predict the class
        predicted_class = model.predict(feature_array)
        st.write(f"Predicted Class: {predicted_class[0]}")
    else:
        st.write("The uploaded CSV does not contain valid data after filtering.")
