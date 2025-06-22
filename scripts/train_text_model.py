import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import joblib
import re
import contractions
from textblob import TextBlob
from pymongo import MongoClient
from datetime import datetime
import uuid
from utils import preprocess_text, get_vader_sentiment
import xgboost as xgb
import json

# Download required NLTK data
print("Downloading required NLTK data...")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'IMDB Dataset.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

def connect_to_mongodb():
    """Connect to MongoDB and return the database object."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['sentiment_analysis']
        return db
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        return None

def validate_data(df):
    """Validate the input data."""
    required_columns = ['review', 'sentiment']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    if df['sentiment'].nunique() != 2:
        raise ValueError("Sentiment column must contain exactly 2 unique values")
    
    if df['review'].isnull().any():
        print("Warning: Found null values in review column. These will be filled with empty strings.")
        df['review'] = df['review'].fillna('')
    
    return df

def save_to_mongodb(db, metrics):
    """Save model metrics to MongoDB."""
    if db is None:
        return
    
    try:
        collection = db['model_metrics']
        collection.insert_one(metrics)
        print(f"Saved metrics for XGBoost (version {metrics['version']}) to MongoDB")
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

def enhanced_clean_text(text):
    """Enhanced text cleaning with more sophisticated preprocessing."""
    try:
        text = str(text)
        text = re.sub(r'<.*?>', '', text)
        text = contractions.fix(text)
        text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
        text = text.lower().strip()
        tokens = text.split()
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        return text

def extract_features(df):
    """Extract additional features from the text."""
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['vader_sentiment'] = df['review'].apply(get_vader_sentiment)
    return df

def save_metrics(metrics):
    """Save model metrics to a JSON file."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'r') as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = []
        existing_metrics.append(metrics)
        with open(METRICS_PATH, 'w') as f:
            json.dump(existing_metrics, f, indent=4, default=str)
        print(f"Saved metrics to {METRICS_PATH}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Convert sentiment to numeric
    label_map = {'negative': 0, 'positive': 1}
    df['sentiment'] = df['sentiment'].map(label_map)
    
    print("Cleaning text...")
    df['clean_text'] = df['review'].apply(preprocess_text)
    
    print("Extracting features...")
    df = extract_features(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df[['clean_text', 'vader_sentiment']], df['sentiment'], 
        test_size=0.2, random_state=42
    )
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 3),
        dtype=np.float32
    )
    
    # Fit and transform training data
    X_train_vec = vectorizer.fit_transform(X_train['clean_text'])
    X_test_vec = vectorizer.transform(X_test['clean_text'])
    
    # Add VADER sentiment as a feature
    X_train_vader = np.array(X_train['vader_sentiment']).reshape(-1, 1)
    X_test_vader = np.array(X_test['vader_sentiment']).reshape(-1, 1)
    X_train_combined = sparse.hstack([X_train_vec, X_train_vader])
    X_test_combined = sparse.hstack([X_test_vec, X_test_vader])
    
    print("Training models...")
    # Train XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        tree_method='hist'
    )
    
    model.fit(X_train_combined, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test_combined)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Get detailed metrics from classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save model and vectorizer
    print("\nSaving model...")
    model_data = {
        'model': model,
        'vectorizer': vectorizer,
        'feature_names': list(vectorizer.get_feature_names_out()) + ['vader_sentiment']
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save metrics
    metrics = {
        'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'accuracy': float(accuracy),
        'model_type': 'XGBoost',
        'model_path': MODEL_PATH,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'vectorizer_features': 5000,
            'ngram_range': (1, 3)
        },
        'class_metrics': {
            'negative': {
                'precision': float(report['0']['precision']),
                'recall': float(report['0']['recall']),
                'f1_score': float(report['0']['f1-score'])
            },
            'positive': {
                'precision': float(report['1']['precision']),
                'recall': float(report['1']['recall']),
                'f1_score': float(report['1']['f1-score'])
            }
        },
        'overall_metrics': {
            'macro_avg_precision': float(report['macro avg']['precision']),
            'macro_avg_recall': float(report['macro avg']['recall']),
            'macro_avg_f1': float(report['macro avg']['f1-score']),
            'weighted_avg_precision': float(report['weighted avg']['precision']),
            'weighted_avg_recall': float(report['weighted avg']['recall']),
            'weighted_avg_f1': float(report['weighted avg']['f1-score'])
        },
        'training_metadata': {
            'train_size': X_train_combined.shape[0],
            'test_size': X_test_combined.shape[0],
            'feature_count': X_train_combined.shape[1],
            'vectorizer_params': {
                'max_features': 5000,
                'ngram_range': (1, 3),
                'min_df': 5,
                'max_df': 0.9
            }
        }
    }
    save_metrics(metrics)

if __name__ == "__main__":
    main()
