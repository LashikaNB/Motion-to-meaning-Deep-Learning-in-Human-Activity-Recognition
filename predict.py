import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
from PIL import Image

# Set page title
st.set_page_config(page_title="Human Motion Recognition", layout="wide")

st.title("Human Activity Recognition")
st.write("Upload an image to classify human motion")

# Load model and class names
@st.cache_resource
def load_model_and_classes():
    try:
        model = load_model("model.keras")
        with open("class_names.json", "r") as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model or class names: {e}")
        return None, None

model, class_names = load_model_and_classes()

# Preprocess function
def preprocess_user_image(img):
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img) / 255.0
    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    # Handle RGBA images
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    return np.expand_dims(img_array, axis=0)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, width=300)
    
    # Make prediction
    try:
        img_input = preprocess_user_image(img)
        pred_probs = model.predict(img_input)[0]
        
        # Get top prediction
        top_idx = np.argmax(pred_probs)
        top_class = class_names[top_idx]
        top_prob = pred_probs[top_idx]
        
        with col2:
            st.subheader("Prediction Results")
            st.success(f"**Predicted Motion: {top_class}**")
            st.info(f"Confidence: {top_prob:.2%}")
        
        # Create probability bar chart
        st.subheader("Class Probabilities")
        fig, ax = plt.subplots(figsize=(7, 4))
        
        # Sort probabilities for better visualization
        sorted_idx = np.argsort(pred_probs)
        sorted_classes = [class_names[i] for i in sorted_idx]
        sorted_probs = [pred_probs[i] for i in sorted_idx]
        
        # Highlight the predicted class
        colors = ['#1f77b4'] * len(sorted_classes)
        highlight_idx = sorted_classes.index(top_class)
        colors[highlight_idx] = '#ff7f0e'  # Highlight color for predicted class
        
        bars = ax.barh(sorted_classes, sorted_probs, color=colors)
        
        # Add percentage text to bars
        for bar in bars:
            width = bar.get_width()
            if width > 0.01:  # Only add text if probability > 1%
                ax.text(max(width + 0.01, 0.05), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.2%}', 
                        va='center')
        
        ax.set_xlabel('Probability')
        ax.set_title('Prediction Probabilities')
        ax.set_xlim(0, 1)
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.info("Please upload an image to get predictions")

# Add some information about the model
with st.expander("About this model"):
    st.write("""
    This model classifies human motions into various categories using MobileNet architecture. 
    
    **Model Details:**
    - Base architecture: MobileNet (pre-trained on ImageNet)
    - Transfer learning approach: Initial training with frozen weights, followed by fine-tuning
    - Fine-tuning strategy: Select par
    ameters were made trainable with extended epochs
    - Training results: Achieved ~96.5% validation accuracy
    
    **Instructions:**
    1. Upload an image showing a person performing an action
    2. The model will analyze the image and predict the motion category
    3. View the prediction confidence and probability distribution
    """)