import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import io
import base64
import plotly.graph_objects as go

# Function to load the trained model from GitHub using the API
def load_model_from_github_api(owner, repo, path):
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        file_content = response.json()['content']
        file_content_decoded = base64.b64decode(file_content)  # Decode the base64 content
        model = joblib.load(io.BytesIO(file_content_decoded))
        return model
    else:
        st.error("Failed to load model from GitHub.")
        return None

# Function to load and filter the new CSV file
def load_and_filter_csv(file, filter_column='L/O', filter_threshold=0.4):
    df = pd.read_csv(file)
    df_filtered = df[df[filter_column] >= filter_threshold]  # Apply the filtering based on L/O column
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

# Function to visualize the segments with predicted categories
def plot_segments(df_filtered, predictions, segment_size):
    # Create the colors for the segments based on the number of segments
    unique_segments = len(predictions)
    color_map = plotly.colors.sequential.Viridis[:unique_segments]

    # Create Plotly figures for NIR and VIS
    fig_nir = go.Figure()
    fig_vis = go.Figure()

    # Plot NIR segments
    for i, (start, pred) in enumerate(zip(range(0, len(df_filtered), segment_size), predictions)):
        end = min(start + segment_size, len(df_filtered))
        # Add NIR line plot
        fig_nir.add_trace(go.Scatter(
            x=df_filtered.index[start:end],
            y=df_filtered['NIR'].iloc[start:end],
            mode='lines',
            line=dict(color=color_map[i]),
            name=f'Segment {i + 1}: Class {pred}'
        ))
        # Add a vertical dashed line to indicate segment boundaries
        fig_nir.add_shape(type="line",
            x0=df_filtered.index[start],
            y0=df_filtered['NIR'].min(),
            x1=df_filtered.index[start],
            y1=df_filtered['NIR'].max(),
            line=dict(color="gray", width=1, dash="dash"))

    fig_nir.update_layout(title='NIR Signal Segmentation with Predicted Categories',
                           xaxis_title='Sample Index',
                           yaxis_title='NIR Value')

    # Plot VIS segments
    for i, (start, pred) in enumerate(zip(range(0, len(df_filtered), segment_size), predictions)):
        end = min(start + segment_size, len(df_filtered))
        # Add VIS line plot
        fig_vis.add_trace(go.Scatter(
            x=df_filtered.index[start:end],
            y=df_filtered['VIS'].iloc[start:end],
            mode='lines',
            line=dict(color=color_map[i]),
            name=f'Segment {i + 1}: Class {pred}'
        ))
        # Add a vertical dashed line to indicate segment boundaries
        fig_vis.add_shape(type="line",
            x0=df_filtered.index[start],
            y0=df_filtered['VIS'].min(),
            x1=df_filtered.index[start],
            y1=df_filtered['VIS'].max(),
            line=dict(color="gray", width=1, dash="dash"))

    fig_vis.update_layout(title='VIS Signal Segmentation with Predicted Categories',
                           xaxis_title='Sample Index',
                           yaxis_title='VIS Value')

    return fig_nir, fig_vis

# Set the GitHub repository details
owner = "meliaph-monitech"
repo = "HyundaiMOBISClassification"
path = "laser_welding_model.joblib"

# Load the trained model from GitHub
model = load_model_from_github_api(owner, repo, path)

# Streamlit app layout
st.title("Laser Welding Signal Classification")
st.write("Upload your CSV file for classification:")

# Step 2: Upload the new CSV file for classification
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
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
