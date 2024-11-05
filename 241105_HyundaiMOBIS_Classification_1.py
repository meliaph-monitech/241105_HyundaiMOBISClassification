import streamlit as st
import zipfile
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Set filter configurations
FILTER_COLUMN = 'L/O'
FILTER_THRESHOLD = 0.4

def load_and_filter_csv(file):
    df = pd.read_csv(file)
    df_filtered = df[df[FILTER_COLUMN] >= FILTER_THRESHOLD]
    return df_filtered[['NIR', 'VIS']]

def process_zip_file(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('temp_extracted')
        folders = [os.path.join('temp_extracted', folder) for folder in os.listdir('temp_extracted') if os.path.isdir(os.path.join('temp_extracted', folder))]
        
        data, labels = [], []
        for folder in folders:
            files = os.listdir(folder)
            label = os.path.basename(folder)
            for file in files:
                filepath = os.path.join(folder, file)
                df_filtered = load_and_filter_csv(filepath)
                data.append(df_filtered.values)
                labels += [label] * len(df_filtered)
                
        data = np.vstack(data)
        labels = np.array(labels)
        
        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.5, stratify=labels)
        return train_data, test_data, train_labels, test_labels

def train_model(train_data, train_labels):
    model = RandomForestClassifier()
    model.fit(train_data, train_labels)
    return model

def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    return classification_report(test_labels, predictions, output_dict=True), confusion_matrix(test_labels, predictions)

def plot_confusion_matrix(cm, labels):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        hoverinfo="text",
        showscale=True
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    st.plotly_chart(fig)

def plot_class_distribution(data, labels):
    df = pd.DataFrame(data, columns=['NIR', 'VIS'])
    df['Label'] = labels
    fig = px.scatter(df, x='NIR', y='VIS', color='Label', title="Class Distribution",
                     labels={'NIR': 'NIR Value', 'VIS': 'VIS Value'},
                     hover_data={'NIR': True, 'VIS': True, 'Label': True})
    st.plotly_chart(fig)

def plot_new_data_classification(new_data, model, train_data, train_labels):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(np.vstack((train_data, new_data)))
    reduced_train, reduced_new = reduced_data[:len(train_data)], reduced_data[len(train_data):]

    # Create DataFrame for visualization
    df_train = pd.DataFrame(reduced_train, columns=['PC1', 'PC2'])
    df_train['Label'] = train_labels
    df_new = pd.DataFrame(reduced_new, columns=['PC1', 'PC2'])
    df_new['Label'] = ['New Data'] * len(df_new)

    fig = px.scatter(df_train, x='PC1', y='PC2', color='Label', title="New Data in Feature Space",
                     hover_data={'PC1': True, 'PC2': True, 'Label': True},
                     labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
    fig.add_trace(go.Scatter(x=df_new['PC1'], y=df_new['PC2'], mode='markers', marker=dict(color='black', symbol='x', size=12),
                             name="New Data", text="New Data"))

    st.plotly_chart(fig)

st.title("Laser Welding Data Classification")

# Step 1: Upload ZIP for Training
uploaded_zip = st.file_uploader("Upload ZIP File", type="zip")

if uploaded_zip:
    train_data, test_data, train_labels, test_labels = process_zip_file(uploaded_zip)
    model = train_model(train_data, train_labels)
    
    # Plot class distribution of training data
    st.subheader("Training and Testing Data Class Distribution")
    plot_class_distribution(np.vstack((train_data, test_data)), np.concatenate((train_labels, test_labels)))
    
    # Evaluate and visualize model performance
    report, cm = evaluate_model(model, test_data, test_labels)
    st.write("Model Evaluation Report:")
    st.json(report)
    st.subheader("Confusion Matrix")
    plot_confusion_matrix(cm, labels=np.unique(train_labels))

# Step 2: Upload Single CSV for Prediction
uploaded_csv = st.file_uploader("Upload CSV for Prediction", type="csv")

if uploaded_csv:
    new_data = load_and_filter_csv(uploaded_csv)
    prediction = model.predict(new_data)
    predicted_class = prediction[0]
    st.write(f"Predicted Category: {predicted_class}")
    
    # Visualize where new data lies in feature space relative to train data
    st.subheader("New Data Position in Feature Space")
    plot_new_data_classification(new_data.values, model, train_data, train_labels)
