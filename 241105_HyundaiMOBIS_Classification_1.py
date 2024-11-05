import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px  # Import Plotly Express for color scales

# Function to load the trained model
def load_model(model_path):
    return joblib.load(model_path)

# Function to load and filter the new CSV file
def load_and_filter_csv(file, filter_column='L/O', filter_threshold=0.4):
    df = pd.read_csv(file)
    df_filtered = df[df[filter_column] >= filter_threshold]  # Apply filtering
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
    fig_nir = go.Figure()
    fig_vis = go.Figure()
    
    # Create a color map for predictions
    unique_labels = np.unique(predictions)
    colors = px.colors.qualitative.Plotly  # Use Plotly's qualitative color palette
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    for i, (start, pred) in enumerate(zip(range(0, len(df_filtered), segment_size), predictions)):
        end = min(start + segment_size, len(df_filtered))
        
        # Use color based on predicted class
        color = color_map[pred]  # Get color for the predicted class
        
        # Plot NIR
        fig_nir.add_trace(go.Scatter(
            x=df_filtered.index[start:end],
            y=df_filtered['NIR'].iloc[start:end],
            mode='lines',
            line=dict(color=color),
            name=f'Segment {i + 1}: Class {pred}'
        ))
        
        # Plot VIS
        fig_vis.add_trace(go.Scatter(
            x=df_filtered.index[start:end],
            y=df_filtered['VIS'].iloc[start:end],
            mode='lines',
            line=dict(color=color),
            name=f'Segment {i + 1}: Class {pred}'
        ))

    fig_nir.update_layout(title='NIR Signal Segmentation',
                          xaxis_title='Sample Index',
                          yaxis_title='NIR Value')
    fig_vis.update_layout(title='VIS Signal Segmentation',
                          xaxis_title='Sample Index',
                          yaxis_title='VIS Value')

    return fig_nir, fig_vis

# Streamlit app layout
st.title("Laser Welding Signal Classification")

# Step 1: Upload the trained model
model_file = st.file_uploader("Upload your trained model file (.joblib):", type='joblib')
if model_file:
    model = load_model(model_file)
    st.success("Model loaded successfully.")

    # Step 2: Upload the new CSV file for classification
    csv_file = st.file_uploader("Upload the new CSV file for classification:", type='csv')
    if csv_file:
        # Load and preprocess the new data
        df_filtered = load_and_filter_csv(csv_file)

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
