# Movie Review Sentiment Analysis

This project implements a sentiment analysis system for movie reviews using machine learning. It can analyze both text and audio reviews to determine if they express positive or negative sentiment.

## Features

- Text-based sentiment analysis
- Audio review transcription and analysis
- Multiple model options:
  - Ensemble (combines multiple models)
  - Naive Bayes
  - Combined Models (Logistic Regression, Random Forest, XGBoost)
- Interactive web interface using Streamlit
- MongoDB integration for model metrics tracking
- Visualization of sentiment trends

## Project Structure

```
movie-review-sentiment-analysis/
├── app.py                 # Streamlit web interface
├── scripts/
│   ├── train_text_model.py    # Model training script
│   ├── split_dataset.py       # Dataset preparation
│   └── generate_audio.py      # Audio processing utilities
├── data/
│   ├── raw/              # Raw dataset
│   ├── processed/        # Processed data
│   └── audio/           # Audio files
├── models/              # Trained models
├── utils.py            # Utility functions
└── requirements.txt    # Project dependencies
```

## Dataset Location

The dataset can be accessed from the following drive folder:  
https://drive.google.com/drive/folders/1z15jcLywsrHIsoUydlCcL_uKUlOeINic?usp=drive_link

**Important:** Before training the model, you must download the dataset from the above location.  
Place the downloaded dataset files into the `data/raw/` directory in the project structure to ensure proper access during training.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd movie-review-sentiment-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

## Usage

1. **Download the dataset** as described in the "Dataset Location" section and place it in the `data/raw/` directory.

2. Train the model:
```bash
python scripts/train_text_model.py
```

3. Run the web interface:
```bash
streamlit run app.py
```

4. Access the web interface at `http://localhost:8501`

## Model Details

The system uses an ensemble approach combining:
- Naive Bayes for text classification
- Combined models (Logistic Regression, Random Forest, XGBoost)
- Feature engineering including:
  - TF-IDF vectorization
  - Text length and word count
  - Sentiment scores
  - Subjectivity analysis

## MongoDB Integration

The system automatically tracks model performance metrics in MongoDB:
- Accuracy
- Precision
- Recall
- F1 Score
- Training metadata
- Model versioning

## Contributing

Feel free to submit issues and enhancement requests!
