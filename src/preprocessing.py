# src/preprocessing.py
import pandas as pd
import re

def preprocess_text(text):
    """Nettoie le texte."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    return text.lower().strip()

def load_and_preprocess_data(file_path, num_positive=5000, num_negative=5000):
    """Charge et nettoie les données."""
    data = pd.read_csv(file_path, encoding='latin1', header=None)
    data.columns = ["sentiment", "id", "date", "query", "user", "TweetText"]
    data['sentiment'] = data['sentiment'].replace(4, 1)  # Convertir 4 en 1 pour positif
    positive_data = data[data['sentiment'] == 1].iloc[:num_positive]
    negative_data = data[data['sentiment'] == 0].iloc[:num_negative]
    combined_data = pd.concat([positive_data, negative_data]).sample(frac=1, random_state=42).reset_index(drop=True)
    combined_data['TweetText'] = combined_data['TweetText'].apply(preprocess_text)
    return combined_data
