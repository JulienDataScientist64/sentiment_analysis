import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import gensim.downloader as api
import mlflow
import mlflow.tensorflow
from azureml.core import Workspace, Run
from src.models.common import (
    log_metrics,
    plot_and_log_accuracy_loss,
    log_classification_report,
    plot_and_log_confusion_matrix
)

# Charger le workspace AzureML et configurer MLflow
try:
    ws = Workspace.from_config(path='/content/drive/MyDrive/AzureML/config.json')
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_experiment("SentimentAnalysis_LSTM_FastText")
    print(f"Workspace AzureML chargé : {ws.name}")
except Exception as e:
    print("Erreur de chargement du Workspace AzureML :", e)

def load_fasttext(word_index, embedding_dim=300):
    """
    Charge FastText et crée une matrice d'embeddings.
    Args:
        word_index (dict): Dictionnaire des mots et indices du tokenizer.
        embedding_dim (int): Taille des embeddings FastText.
    Returns:
        np.ndarray: Matrice d'embeddings FastText.
    """
    print("Chargement des embeddings FastText...")
    fasttext = api.load("fasttext-wiki-news-subwords-300")
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in fasttext:
            embedding_matrix[i] = fasttext[word]
    return embedding_matrix

def build_lstm_model(embedding_matrix, input_length, embedding_dim):
    """
    Construit un modèle LSTM avec les embeddings FastText.
    Args:
        embedding_matrix (np.ndarray): Matrice des embeddings FastText.
        input_length (int): Longueur des séquences d'entrée.
        embedding_dim (int): Dimension des embeddings.
    Returns:
        Sequential: Modèle Keras.
    """
    print("Construction du modèle LSTM avec FastText...")
    model = Sequential([
        Embedding(input_dim=embedding_matrix.shape[0],
                  output_dim=embedding_dim,
                  weights=[embedding_matrix],
                  input_length=input_length,
                  trainable=False),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm_fasttext(file_path, epochs=5, batch_size=32):
    """
    Entraîne un modèle LSTM avec FastText.
    Args:
        file_path (str): Chemin vers le fichier contenant les données.
        epochs (int): Nombre d'époques pour l'entraînement.
        batch_size (int): Taille des lots pour l'entraînement.
    """
    # Étape 1 : Charger et prétraiter les données
    print("Chargement des données brutes...")
    data = pd.read_csv(file_path, encoding='latin1', header=None)
    data.columns = ["sentiment", "id", "date", "query", "user", "TweetText"]
    data['sentiment'] = data['sentiment'].replace(4, 1)  # Convertir 4 en 1 pour les positifs

    print("Prétraitement et équilibrage des données...")
    pos_data = data[data['sentiment'] == 1].sample(10000, random_state=42)
    neg_data = data[data['sentiment'] == 0].sample(10000, random_state=42)
    df = pd.concat([pos_data, neg_data]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Étape 2 : Tokenizer et séquences
    print("Tokenisation et création des séquences...")
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(df['TweetText'])
    sequences = tokenizer.texts_to_sequences(df['TweetText'])
    word_index = tokenizer.word_index
    X = pad_sequences(sequences, maxlen=100)
    y = df['sentiment']

    # Sauvegarder le tokenizer
    tokenizer_path = "tokenizer.pkl"
    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Tokenizer sauvegardé localement : {tokenizer_path}")

    # Enregistrer le tokenizer comme artefact dans Azure ML
    run = Run.get_context()
    run.upload_file(name="tokenizer.pkl", path_or_stream=tokenizer_path)
    print("Tokenizer enregistré comme artefact dans Azure ML.")

    # Étape 3 : Charger les embeddings FastText
    embedding_dim = 300
    embedding_matrix = load_fasttext(word_index, embedding_dim)

    # Étape 4 : Diviser les données en ensembles d'entraînement et de test
    print("Division des données...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Étape 5 : Construire le modèle
    input_length = X_train.shape[1]
    model = build_lstm_model(embedding_matrix, input_length, embedding_dim)

    # Étape 6 : Activer l'autologging et entraîner
    print("Activation de l'autologging avec MLflow et début de l'entraînement...")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="LSTM_FastText"):
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

        # Journalisation des courbes d'accuracy et de perte
        plot_and_log_accuracy_loss(history, "fasttext")

        # Prédictions
        y_pred = (model.predict(X_test) > 0.5).astype(int)

        # Journaliser la matrice de confusion
        plot_and_log_confusion_matrix(y_test, y_pred, labels=[0, 1], title="Matrice de Confusion - LSTM FastText")

        # Journaliser le rapport de classification
        log_classification_report(y_test, y_pred, target_names=["Négatif", "Positif"], report_title="Rapport de Classification - LSTM FastText")

        print("Entraînement terminé et métriques enregistrées avec MLflow.")

if __name__ == "__main__":
    train_lstm_fasttext(
        file_path="/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/training.csv",
        epochs=5,
        batch_size=32
    )
