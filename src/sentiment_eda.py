import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import networkx as nx
from itertools import combinations
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings

# Configuration
warnings.filterwarnings('ignore')
nltk.download('stopwords')

class SentimentEDA:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def load_data(self):
        """Charger et préparer les données."""
        print("Chargement des données...")
        DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]
        self.data = pd.read_csv(self.file_path, encoding='latin1')
        self.data.columns = DATASET_COLUMNS
        self.data.drop(['ids', 'flag', 'user'], axis=1, inplace=True)
        self.data.drop_duplicates(subset='TweetText', keep='first', inplace=True)
        print(f"Données chargées : {self.data.shape[0]} lignes après suppression des doublons.")

    def clean_tweet(self, tweet):
        """Nettoyer un tweet."""
        tweet = re.sub(r"@\w+", "", tweet)
        tweet = re.sub(r"http\S+", "", tweet)
        tweet = re.sub(r"[^a-zA-Z\s]", " ", tweet)
        tweet = re.sub(r"\s+", " ", tweet).strip()
        words = tweet.lower().split()
        words = [word for word in words if word not in self.stop_words]
        return ' '.join([self.stemmer.stem(word) for word in words])

    def clean_tweets(self):
        """Appliquer le nettoyage à tous les tweets."""
        print("Nettoyage des tweets en cours...")
        if 'TweetText' in self.data.columns:
            self.data['Clean_TweetText'] = self.data['TweetText'].apply(self.clean_tweet)
            print("Nettoyage terminé.")
        else:
            print("La colonne 'TweetText' est absente.")

    def analyze_tweet_lengths(self):
        """Analyser les longueurs des tweets avant et après nettoyage."""
        self.data['Original_Length'] = self.data['TweetText'].str.len()
        self.data['Cleaned_Length'] = self.data['Clean_TweetText'].str.len()

        plt.figure(figsize=(12, 6))
        plt.hist(self.data['Original_Length'], bins=30, alpha=0.6, label='Avant nettoyage')
        plt.hist(self.data['Cleaned_Length'], bins=30, alpha=0.6, label='Après nettoyage')
        plt.title("Distribution des longueurs des tweets")
        plt.xlabel("Longueur")
        plt.ylabel("Fréquence")
        plt.legend()
        plt.show()

    def analyze_time_distribution(self):
        """Analyser la distribution des tweets par jour et heure."""
        if 'date' not in self.data.columns:
            print("Les dates ne sont pas disponibles pour cette analyse.")
            return

        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        self.data['DayOfWeek'] = self.data['date'].dt.day_name()
        self.data['Hour'] = self.data['date'].dt.hour

        plt.figure(figsize=(12, 6))
        self.data.groupby('DayOfWeek')['target'].value_counts().unstack().plot(kind='bar', stacked=True)
        plt.title("Distribution des tweets par jour de la semaine")
        plt.xlabel("Jour de la semaine")
        plt.ylabel("Nombre de tweets")
        plt.show()

        plt.figure(figsize=(12, 6))
        self.data.groupby('Hour')['target'].value_counts().unstack().plot(kind='bar', stacked=True)
        plt.title("Distribution des tweets par heure")
        plt.xlabel("Heure")
        plt.ylabel("Nombre de tweets")
        plt.show()

    def display_sentiment_counts(self):
        """Afficher les chiffres précis des tweets positifs et négatifs."""
        positive_count = self.data[self.data['target'] == 4].shape[0]
        negative_count = self.data[self.data['target'] == 0].shape[0]
        print(f"Nombre de tweets positifs : {positive_count}")
        print(f"Nombre de tweets négatifs : {negative_count}")

    def perform_eda(self):
        """Réaliser l'EDA complet."""
        print("Début de l'EDA...")
        self.clean_tweets()
        self.analyze_tweet_lengths()
        self.analyze_time_distribution()
        self.display_sentiment_counts()
        print("EDA terminé.")

    def save_cleaned_data(self, output_path):
        """Sauvegarder les données nettoyées."""
        self.data.to_csv(output_path, index=False)
        print(f"Données nettoyées enregistrées dans {output_path}.")

if __name__ == "__main__":
    eda = SentimentEDA(file_path='/content/drive/MyDrive/sentiment_analysis/training.csv')
    eda.load_data()
    eda.perform_eda()