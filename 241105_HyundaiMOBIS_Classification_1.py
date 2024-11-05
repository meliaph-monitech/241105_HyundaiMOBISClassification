# Importing required libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.graph_objs as go

# Function to load the trained model
def load_model(model_path):
    return joblib.load(model_path)

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

# Function to visualize the segments with predicted categories using Plotly
def plot_segments(df_filtered, predictions, segment_size):
    fig_nir = go.Figure()
    fig_vis = go.Figure()

    colors = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red'}  # Define colors for each class

    for i, (start, pred) in enumerate(zip(range(0, len(df_filtered), segment_size), predictions)):
        end = min(start + segment_size, len(df_filtered))
        segment_nir = df_filtered['NIR'].iloc[start:end]
        segment_vis = df_filtered['VIS'].iloc[start:end]
        
        fig_nir.add_trace(go.Scatter(
            x=np.arange(start, end),
            y=segment_nir,
            mode='lines',
            line=dict(color=colors[pred]),
            name=f'Segment {i+1}: Class {pred}'
        ))

        fig_vis.add_trace(go.Scatter(
            x=np.arange(start, end),
            y=segment_vis,
            mode='lines',
            line=dict(color=colors[pred]),
            name=f'Segment {i+1}: Class {pred}'
        ))

    fig_nir.update_layout(title='NIR Signal Segmentation with Predicted Categories',
                           xaxis_title='Sample Index',
                           yaxis_title='NIR Value')
    
    fig_vis.update_layout(title='VIS Signal Segmentation with Predicted Categories',
                           xaxis_title='Sample Index',
                           yaxis_title='VIS Value')

    return fig_nir, fig_vis

# Streamlit App
st.title('Laser Welding Signal Classification')
st.write("Upload your trained model (.joblib) and the new CSV file for classification.")

# Step 1: Upload the trained model
uploaded_model = st.file_uploader("Upload your trained model file (.joblib):", type='joblib')
if uploaded_model is not None:
    model = load_model(uploaded_model)
    st.success("Model uploaded successfully!")

    # Step 2: Upload the new CSV file for classification
    uploaded_csv = st.file_uploader("Upload the new CSV file for classification:", type='csv')
    if uploaded_csv is not None:
        df_filtered = load_and_filter_csv(uploaded_csv)

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

            # Show the plots in the Streamlit app
            st.plotly_chart(fig_nir)
            st.plotly_chart(fig_vis)
        else:
            st.warning("The uploaded CSV does not contain valid data after filtering.")
