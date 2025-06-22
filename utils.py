import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import speech_recognition as sr
import os
from pydub import AudioSegment
import re
import contractions
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

NEGATION_PATTERNS = [
    r'(not\s+too\s+\w+)',
    r'(not\s+very\s+\w+)',
    r'(not\s+at\s+all\s+\w+)',
    r'(not\s+\w+)',
    r'(never\s+\w+)',
    r'(no\s+\w+)'
]

VADER = SentimentIntensityAnalyzer()

def handle_negations(text):
    for pattern in NEGATION_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            joined = '_'.join(match.split())
            text = text.replace(match, joined)
    return text

def get_vader_sentiment(text):
    if not isinstance(text, str):
        return 0.0
    return VADER.polarity_scores(text)['compound']

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = handle_negations(text)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    text = ' '.join(tokens)
    return text

def audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext == '.mp3':
        sound = AudioSegment.from_mp3(audio_path)
        wav_path = audio_path.replace('.mp3', '.wav')
        sound.export(wav_path, format='wav')
        audio_path = wav_path
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Could not request results; {e}"

def get_sentiment_score(text):
    return TextBlob(text).sentiment.polarity 