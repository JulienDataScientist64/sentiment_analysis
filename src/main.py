# src/main.py

import sys
import os  # Ajout de l'import manquant

# Ajout du chemin parent de 'src' pour éviter la redondance 'src.models'
sys.path.insert(0, '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning')

import argparse
from src.models.common import log_metrics
from src.models.tfidf_logistic_regression import train_tfidf_logistic_regression
from src.models.lstm_word2vec import train_lstm_word2vec
from src.models.lstm_fasttext import train_lstm_fasttext
from src.models.bert_cnn import train_with_bert_cnn
from src.models.distilbert_cnn import train_with_distilbert_cnn  # Assurez-vous que cette fonction est correctement nommée

def parse_args():
    """
    Parse les arguments pour exécuter un modèle spécifique.
    """
    parser = argparse.ArgumentParser(description="Lancer un entraînement pour un modèle spécifique.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tfidf", "lstm_word2vec", "lstm_fasttext", "bert_cnn", "distilbert_cnn"],  # Ajout de 'bert_cnn'
        help="Nom du modèle à exécuter (tfidf, lstm_word2vec, lstm_fasttext, bert_cnn, distilbert_cnn)."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/training.csv",
        help="Chemin vers le fichier CSV contenant les données d'entraînement."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Nombre d'époques pour l'entraînement (applicable aux modèles LSTM et DistilBERT)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Taille des lots pour l'entraînement (applicable aux modèles LSTM et DistilBERT)."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Longueur maximale des séquences (applicable au modèle DistilBERT)."
    )
    return parser.parse_args()


def main():
    """
    Point d'entrée principal pour exécuter un modèle en fonction des arguments fournis.
    """
    args = parse_args()

    # Vérification du fichier d'entrée
    if not os.path.exists(args.file_path):
        raise FileNotFoundError(f"Fichier non trouvé : {args.file_path}")

    # Appeler le bon modèle en fonction des arguments
    if args.model == "tfidf":
        print("Entraînement du modèle TF-IDF Logistic Regression...")
        train_tfidf_logistic_regression(file_path=args.file_path)
    elif args.model == "lstm_word2vec":
        print("Entraînement du modèle LSTM avec Word2Vec...")
        train_lstm_word2vec(file_path=args.file_path, epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == "lstm_fasttext":
        print("Entraînement du modèle LSTM avec FastText...")
        train_lstm_fasttext(file_path=args.file_path, epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == "bert_cnn":
        print("Entraînement du modèle BERT avec CNN...")
        train_with_bert_cnn(file_path=args.file_path, epochs=args.epochs, batch_size=args.batch_size)
    elif args.model == "distilbert_cnn":
        print("Entraînement du modèle DistilBERT avec CNN...")
        train_with_distilbert_cnn(file_path=args.file_path, max_len=args.max_len, epochs=args.epochs, batch_size=args.batch_size)
    else:
        print(f"Modèle non pris en charge : {args.model}")


if __name__ == "__main__":
    main()
