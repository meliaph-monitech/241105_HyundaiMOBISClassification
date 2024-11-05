import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go

# Function to load the trained model from GitHub
def load_model_from_github(model_url):
    response = requests.get(model_url)
    model = joblib.load(io.BytesIO(response.content))
    return model

# Function to load and filter the new CSV file
def load_and_filter_csv(file, filter_column='L/O', filter_threshold=0.4):
    df = pd.read_csv(file)
    df_filtered = df[df[filter_column] >= filter_threshold]  # Apply filtering based on L/O column
    return df_filtered[['NIR', 'VIS']]

# Function to segment the signal data
def segment_data(df_filtered, segment_size=10000):
    segments = []
    for start in range(0, len(df_filtered), segment_size):
        end = min(start + segment_size, len(df_filtered))
        segment = df_filtered.iloc[start:end]
        if len(segment) > 0:
            segments.append(segment)
    return segments

# Function to extract features from the new data
def extract_features(df_segment):
    return {
        'mean_NIR': np.mean(df_segment['NIR']),
        'std_NIR': np.std(df_segment['NIR']),
        'max_NIR': np.max(df_segment['NIR']),
        'min_NIR': np.min(df_segment['NIR']),
        'mean_VIS': np.mean(df_segment['VIS']),
        'std_VIS': np.std(df_segment['VIS']),
        'max_VIS': np.max(df_segment['VIS']),
        'min_VIS': np.min(df_segment['VIS']),
    }

# Function to plot the segments with predicted categories
def plot_segments(df_filtered, predictions, segment_size=10000):
    # Create color mapping for segments
    unique_segments = len(predictions)
    colors = plotly.colors.sequential.Viridis[:unique_segments]  # Use Plotly's Viridis color map

    # Create subplots for NIR and VIS
    fig_nir = go.Figure()
    fig_vis = go.Figure()

    # Plot NIR signal
    for i, (start, pred) in enumerate(zip(range(0, len(df_filtered), segment_size), predictions)):
        end = min(start + segment_size, len(df_filtered))
        fig_nir.add_trace(go.Scatter(
            x=df_filtered.index[start:end],
            y=df_filtered['NIR'].iloc[start:end],
            mode='lines',
            line=dict(color=colors[i % unique_segments]),  # Color by segment index
            name=f'Segment {i+1}: Class {pred}'
        ))
        
        # Add vertical dashed lines to indicate segment boundaries
        if end < len(df_filtered):
            fig_nir.add_vline(x=end, line=dict(dash='dash', color='gray'))

    fig_nir.update_layout(title='NIR Signal Segmentation with Predicted Categories',
                          xaxis_title='Sample Index',
                          yaxis_title='NIR Signal Value')

    # Plot VIS signal
    for i, (start, pred) in enumerate(zip(range(0, len(df_filtered), segment_size), predictions)):
        end = min(start + segment_size, len(df_filtered))
        fig_vis.add_trace(go.Scatter(
            x=df_filtered.index[start:end],
            y=df_filtered['VIS'].iloc[start:end],
            mode='lines',
            line=dict(color=colors[i % unique_segments]),  # Color by segment index
            name=f'Segment {i+1}: Class {pred}'
        ))
        
        # Add vertical dashed lines to indicate segment boundaries
        if end < len(df_filtered):
            fig_vis.add_vline(x=end, line=dict(dash='dash', color='gray'))

    fig_vis.update_layout(title='VIS Signal Segmentation with Predicted Categories',
                          xaxis_title='Sample Index',
                          yaxis_title='VIS Signal Value')

    return fig_nir, fig_vis

# Set the GitHub URL for the model
model_url = "https://raw.githubusercontent.com/meliaph-monitech/HyundaiMOBISClassification/main/laser_welding_model.joblib"

# Load the trained model
model = load_model_from_github(model_url)

# Upload the new CSV file for classification
uploaded_file = st.file_uploader("Upload your new CSV file for classification", type='csv')

if uploaded_file is not None:
    # Load and preprocess the new data
    df_filtered = load_and_filter_csv(uploaded_file)

    if not df_filtered.empty:
        # Segment the filtered data
        segments = segment_data(df_filtered)

        predictions = []
        for segment in segments:
            # Extract features for each segment
            features = extract_features(segment)
            feature_array = np.array(list(features.values())).reshape(1, -1)  # Reshape for model input

            # Predict the class
            predicted_class = model.predict(feature_array)
            predictions.append(predicted_class[0])  # Collect predictions for each segment

        # Visualize the segments with predicted categories
        fig_nir, fig_vis = plot_segments(df_filtered, predictions, segment_size=10000)
        st.plotly_chart(fig_nir)
        st.plotly_chart(fig_vis)
    else:
        st.warning("The uploaded CSV does not contain valid data after filtering.")
