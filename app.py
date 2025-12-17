"""
Indoor Plant Disease Detection Dashboard
Powered by EfficientNet-B0 Transfer Learning
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# Page Configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="Leaf",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        background-color: #f0f8f0;
        margin: 10px 0;
    }
    .out-of-scope-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #FF5722;
        background-color: #fff3f3;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Configuration (Relative paths for Docker)
MODEL_PATH = "models/efficientnetb0_plant_disease_final.keras"
CLASS_MAPPINGS_PATH = "data/class_mappings.json"
METRICS_PATH = "data/per_class_metrics.csv"
IMG_SIZE = (224, 224)

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.6
ENTROPY_THRESHOLD = 2.5
TOP2_RATIO_THRESHOLD = 2.0

# ==================== LOAD MODEL & DATA ====================

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_class_mappings():
    try:
        with open(CLASS_MAPPINGS_PATH, 'r') as f:
            data = json.load(f)
        return data['class_names'], data['class_indices']
    except Exception as e:
        st.error(f"Error loading class mappings: {e}")
        return None, None

@st.cache_data
def load_metrics():
    try:
        if os.path.exists(METRICS_PATH):
            return pd.read_csv(METRICS_PATH)
        return None
    except Exception as e:
        st.warning(f"Metrics file not found: {e}")
        return None

# Load resources
model = load_model()
class_names, class_indices = load_class_mappings()

if model is None or class_names is None:
    st.error("Failed to load model or class mappings. Please check file paths.")
    st.stop()

idx_to_class = {v: k for k, v in class_indices.items()}
metrics_df = load_metrics()

# ==================== PREDICTION FUNCTIONS ====================

def preprocess_image(image):
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def calculate_entropy(probabilities):
    return -np.sum(probabilities * np.log(probabilities + 1e-10))

def is_out_of_scope(probabilities, conf_threshold=0.6):
    top_indices = np.argsort(probabilities)[::-1]
    top_conf = probabilities[top_indices[0]]
    second_conf = probabilities[top_indices[1]]
    
    entropy = calculate_entropy(probabilities)
    top2_ratio = top_conf / (second_conf + 1e-10)
    
    rejection_reasons = []
    
    if top_conf < conf_threshold:
        rejection_reasons.append(f"Low confidence ({top_conf:.2%} < {conf_threshold:.0%})")
    if entropy > ENTROPY_THRESHOLD:
        rejection_reasons.append(f"High uncertainty (entropy: {entropy:.2f})")
    if top2_ratio < TOP2_RATIO_THRESHOLD:
        rejection_reasons.append(f"Ambiguous prediction (similar top-2 probabilities)")
    
    return len(rejection_reasons) > 0, rejection_reasons, {
        'confidence': top_conf,
        'entropy': entropy,
        'top2_ratio': top2_ratio
    }

def predict(image, conf_threshold):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    
    is_rejected, rejection_reasons, metrics = is_out_of_scope(predictions, conf_threshold)
    
    top_indices = np.argsort(predictions)[::-1]
    top_predictions = [
        {
            'class': idx_to_class[i],
            'probability': float(predictions[i]),
            'percentage': f"{predictions[i]*100:.2f}%"
        }
        for i in top_indices[:5]
    ]
    
    result = {
        'is_out_of_scope': is_rejected,
        'rejection_reasons': rejection_reasons,
        'metrics': metrics,
        'top_predictions': top_predictions,
        'predicted_class': top_predictions[0]['class'],
        'confidence': top_predictions[0]['probability'],
        'percentage': top_predictions[0]['percentage']  # Fixed
    }
    return result

# ==================== UI COMPONENTS ====================

def display_header():
    st.markdown('<p class="main-header">Leaf Plant Disease Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Disease Classification for Indoor Plants</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", "EfficientNet-B0", "Transfer Learning")
    with col2:
        st.metric("Classes", "23", "Disease Types")
    with col3:
        st.metric("Test Accuracy", "99.02%", "+2.93%")
    with col4:
        st.metric("Top-3 Accuracy", "100%", "Perfect")

def display_prediction_result(result, image):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if result['is_out_of_scope']:
            st.markdown('<div class="out-of-scope-box">', unsafe_allow_html=True)
            st.markdown("### Out-of-Scope Detection")
            st.warning("This image does not match any known plant disease classes.")
            st.markdown("**Rejection Reasons:**")
            for reason in result['rejection_reasons']:
                st.markdown(f"- {reason}")
            st.markdown("**Metrics:**")
            st.write(f"- Max Confidence: {result['metrics']['confidence']:.2%}")
            st.write(f"- Entropy: {result['metrics']['entropy']:.2f}")
            st.write(f"- Top-2 Ratio: {result['metrics']['top2_ratio']:.2f}")
            st.markdown("**Top Predictions (if forced to classify):**")
            for i, pred in enumerate(result['top_predictions'][:3], 1):
                st.write(f"{i}. {pred['class']}: {pred['percentage']}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Prediction Results")
            st.success(f"**Detected Disease:** {result['predicted_class']}")
            st.metric("Confidence", result['percentage'], delta=f"Entropy: {result['metrics']['entropy']:.2f}")
            
            plant_species = result['predicted_class'].split('_')[0]
            disease_name = '_'.join(result['predicted_class'].split('_')[1:]).replace('_', ' ')
            st.info(f"**Plant Species:** {plant_species}")
            st.info(f"**Disease:** {disease_name}")
            
            # Confidence bar
            confidence_color = "#4CAF50" if result['confidence'] > 0.9 else "#FF9800" if result['confidence'] > 0.7 else "#F44336"
            st.markdown(f"""
                <div style="background-color: #e0e0e0; border-radius: 5px; height: 30px; width: 100%;">
                    <div style="background-color: {confidence_color}; height: 100%; width: {result['confidence']*100}%; 
                         border-radius: 5px; text-align: center; line-height: 30px; color: white; font-weight: bold;">
                        {result['percentage']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top 5 Chart
            st.markdown("### Top 5 Predictions")
            fig = go.Figure(data=[
                go.Bar(
                    y=[pred['class'] for pred in result['top_predictions']],
                    x=[pred['probability'] for pred in result['top_predictions']],
                    orientation='h',
                    marker=dict(color=[pred['probability'] for pred in result['top_predictions']], colorscale='Greens'),
                    text=[pred['percentage'] for pred in result['top_predictions']],
                    textposition='auto',
                )
            ])
            fig.update_layout(xaxis_title="Probability", yaxis_title="Disease Class", height=300, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Class Metrics
            if metrics_df is not None:
                class_metric = metrics_df[metrics_df['Class'] == result['predicted_class']]
                if not class_metric.empty:
                    st.markdown("### Model Performance for This Class")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Precision", f"{class_metric.iloc[0]['Precision']:.2%}")
                    with c2:
                        st.metric("Recall", f"{class_metric.iloc[0]['Recall']:.2%}")
                    with c3:
                        st.metric("F1-Score", f"{class_metric.iloc[0]['F1-Score']:.2%}")

# ==================== MAIN APP ====================

def main():
    display_header()
    
    conf_threshold = 0.6  # Default
    
    with st.sidebar:
        st.markdown("## Settings")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.05, help="Minimum confidence to accept prediction")
        st.info(f"Threshold set to: {conf_threshold:.0%}")
        st.markdown("---")
        st.markdown("## About")
        st.info("""
        Detects 23 diseases across 5 indoor plants using EfficientNet-B0.
        - 30,748 training images
        - Test Accuracy: 99.02%
        """)
        st.markdown("## Detection Modes")
        st.write("**In-Scope:** Known diseases")
        st.write("**Out-of-Scope:** Unknown plants/diseases")

    st.markdown("## Upload Plant Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                with st.spinner('Analyzing image...'):
                    result = predict(image, conf_threshold)
                display_prediction_result(result, image)
                
                # Download Results
                st.markdown("---")
                result_text = f"""Plant Disease Detection Results
================================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Status: {'Out-of-Scope' if result['is_out_of_scope'] else 'In-Scope'}
Predicted Class: {result['predicted_class']}
Confidence: {result['confidence']:.2%}

Top 5 Predictions:
"""
                for i, pred in enumerate(result['top_predictions'], 1):
                    result_text += f"{i}. {pred['class']}: {pred['percentage']}\n"
                
                if result['is_out_of_scope']:
                    result_text += "\nRejection Reasons:\n"
                    for reason in result['rejection_reasons']:
                        result_text += f"- {reason}\n"
                
                st.download_button(
                    label="Download Results",
                    data=result_text,
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Please upload an image to get started")
            st.markdown("### Example Use Cases")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**In-Scope:**\n- Rose black spot\n- Hibiscus blight")
            with c2:
                st.markdown("**Out-of-Scope:**\n- Aloe vera\n- Cactus")
            with c3:
                st.markdown("**Tips:**\n- Clear image\n- Good lighting")

    with col2:
        if metrics_df is not None:
            st.markdown("### Top 10 Classes (F1-Score)")
            st.dataframe(
                metrics_df[['Class', 'F1-Score']].sort_values('F1-Score', ascending=False).head(10),
                hide_index=True,
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Leaf Indoor Plant Disease Detection | Powered by EfficientNet-B0</p>
            <p>Built with Streamlit | Accuracy: 99.02%</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()