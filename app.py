import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import preprocess_text, get_sentiment_score, get_vader_sentiment
import os
from scipy import sparse
import tempfile

# Set page config
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('models', 'ensemble_model.pkl')
        if not os.path.exists(model_path):
            st.error("Model file not found. Please train the model first.")
            return None, None
        
        model_data = joblib.load(model_path)
        if isinstance(model_data, dict) and 'model' in model_data and 'vectorizer' in model_data:
            return model_data['model'], model_data['vectorizer']
        else:
            st.error("Invalid model file format.")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model and vectorizer
model, vectorizer = load_model()

def get_confidence_class(confidence):
    if confidence >= 0.8:
        return 'confidence-high'
    elif confidence >= 0.6:
        return 'confidence-medium'
    else:
        return 'confidence-low'

def predict_sentiment(text):
    try:
        if model is None or vectorizer is None:
            return "Error", 0.0, "confidence-low"
        # Preprocess text
        clean = preprocess_text(text)
        # VADER sentiment feature
        vader_score = get_vader_sentiment(text)
        # Transform text
        vec = vectorizer.transform([clean])
        # Combine with VADER feature
        vader_arr = np.array([[vader_score]])
        features = sparse.hstack([vec, vader_arr])
        # Ensure feature dimensions match
        if features.shape[1] != model.n_features_in_:
            st.error(f"Feature dimension mismatch. Model expects {model.n_features_in_}, got {features.shape[1]}. Please retrain the model.")
            return "Error", 0.0, "confidence-low"
        # Get prediction and probability
        pred = model.predict(features)[0]
        proba = np.max(model.predict_proba(features))
        sentiment = 'Positive' if pred == 1 else 'Negative'
        confidence = proba if pred == 1 else 1 - proba
        confidence_class = get_confidence_class(confidence)
        return sentiment, confidence, confidence_class
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return "Error", 0.0, "confidence-low"

# Main app
st.markdown('<h1 class="main-header">Movie Review Sentiment Analysis</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Input Type")
input_type = st.sidebar.radio("Choose input type:", ["Text", "Audio"])

# Main content
if input_type == "Text":
    st.write("Enter your movie review below:")
    user_text = st.text_area("Review", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_text.strip() == '':
            st.warning('Please enter a review.')
        else:
            sentiment, confidence, confidence_class = predict_sentiment(user_text)
            st.write(f'**Predicted Sentiment:** {sentiment}')
            st.markdown(f'**Confidence:** <span class="{confidence_class}">{confidence:.2%}</span>', unsafe_allow_html=True)
            
            # Additional analysis
            st.subheader("Additional Analysis")
            sentiment_score = get_sentiment_score(user_text)
            st.write(f"TextBlob Sentiment Score: {sentiment_score:.2f}")
            
            # Word count analysis
            word_count = len(user_text.split())
            st.write(f"Word Count: {word_count}")
            
            # Visualize sentiment distribution
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=['Negative', 'Positive'], 
                       y=[1-confidence, confidence],
                       palette=['#F44336', '#4CAF50'])
            plt.title('Sentiment Distribution')
            plt.ylabel('Probability')
            st.pyplot(fig)

else:  # Audio input
    st.write("Upload an audio file of your review:")
    audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'])
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Transcribe and Analyze"):
            try:
                from utils import audio_to_text
                # Save uploaded file to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(audio_file.read())
                    tmp_path = tmp_file.name
                text = audio_to_text(tmp_path)
                os.remove(tmp_path)
                if text:
                    st.write("Transcribed Text:")
                    st.write(text)
                    sentiment, confidence, confidence_class = predict_sentiment(text)
                    st.write(f'**Predicted Sentiment:** {sentiment}')
                    st.markdown(f'**Confidence:** <span class="{confidence_class}">{confidence:.2%}</span>', unsafe_allow_html=True)
                else:
                    st.error("Could not transcribe audio. Please try again.")
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")