import pandas as pd
import numpy as np
import joblib
import requests
import io
import plotly.graph_objs as go
import streamlit as st

# Function to load the trained model from GitHub
def load_model_from_github(url):
    response = requests.get(url)
    model = joblib.load(io.BytesIO(response.content))
    return model

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

# Function to plot segments with predicted categories
def plot_segments(df_filtered, predictions, segment_size):
    # Create the color map
    unique_segments = len(predictions)
    colors = plotly.colors.sequential.Viridis[:unique_segments]  # Use Plotly's Viridis color map

    # Create the figure for NIR and VIS
    fig_nir = go.Figure()
    fig_vis = go.Figure()

    for i, (start, pred) in enumerate(zip(range(0, len(df_filtered), segment_size), predictions)):
        end = min(start + segment_size, len(df_filtered))
        fig_nir.add_trace(go.Scatter(x=df_filtered.index[start:end], 
                                       y=df_filtered['NIR'].iloc[start:end],
                                       mode='lines',
                                       line=dict(color=colors[i % unique_segments]),
                                       name=f'Segment {i + 1}: Class {pred}'))
        
        fig_vis.add_trace(go.Scatter(x=df_filtered.index[start:end], 
                                      y=df_filtered['VIS'].iloc[start:end],
                                      mode='lines',
                                      line=dict(color=colors[i % unique_segments]),
                                      name=f'Segment {i + 1}: Class {pred}'))
        
        # Add vertical dashed line for segmentation
        if end < len(df_filtered):
            fig_nir.add_vline(x=df_filtered.index[end], line_width=1, line_dash="dash", line_color="gray")
            fig_vis.add_vline(x=df_filtered.index[end], line_width=1, line_dash="dash", line_color="gray")

    fig_nir.update_layout(title='NIR Signal Segmentation with Predicted Categories',
                          xaxis_title='Sample Index',
                          yaxis_title='NIR',
                          showlegend=True)

    fig_vis.update_layout(title='VIS Signal Segmentation with Predicted Categories',
                          xaxis_title='Sample Index',
                          yaxis_title='VIS',
                          showlegend=True)

    return fig_nir, fig_vis

# Streamlit App
st.title("Laser Welding Signal Classification")

# Load the trained model from GitHub
model_url = "https://github.com/meliaph-monitech/HyundaiMOBISClassification/blob/main/laser_welding_model.joblib"
model = load_model_from_github(model_url)

# Step 2: Upload the new CSV file for classification
uploaded_file = st.file_uploader("Upload the new CSV file for classification:", type="csv")

if uploaded_file is not None:
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
