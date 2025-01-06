import os
import pickle
import time
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from gensim.downloader import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --------------------------------------------------
# 1. Gestion des arguments en ligne de commande
# --------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Bidirectional GRU avec embeddings Word2Vec.")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers le fichier CSV contenant les données.")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie pour sauvegarder les modèles et rapports.")
    parser.add_argument("--embedding_dim", type=int, default=100, help="Dimension des embeddings Word2Vec.")
    parser.add_argument("--max_features", type=int, default=10000, help="Nombre maximal de mots pour le tokenizer.")
    parser.add_argument("--max_len", type=int, default=50, help="Longueur maximale des séquences.")
    return parser.parse_args()

# --------------------------------------------------
# 2. Fonction principale
# --------------------------------------------------
def main():
    args = parse_arguments()

    # Vérification GPU
    device_name = tf.test.gpu_device_name()
    if device_name == '/device:GPU:0':
        print(f"[INFO] GPU détecté: {device_name}")
        tf.config.optimizer.set_jit(True)  # Active XLA
    else:
        print("[WARNING] Aucun GPU détecté, exécution sur CPU.")

    # Chargement des données
    print(f"Chargement des données depuis {args.data_path}...")
    balanced_data = pd.read_csv(args.data_path)[['target', 'TweetText']]

    # Split des données
    train_data, temp_data = train_test_split(
        balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['target']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=temp_data['target']
    )

    # Tokenisation et séquences
    tokenizer = Tokenizer(num_words=args.max_features)
    tokenizer.fit_on_texts(train_data['TweetText'])

    def preprocess_texts(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=args.max_len)

    X_train = preprocess_texts(train_data['TweetText'])
    X_val = preprocess_texts(val_data['TweetText'])
    X_test = preprocess_texts(test_data['TweetText'])
    y_train = train_data['target'].values
    y_val = val_data['target'].values
    y_test = test_data['target'].values

    # Chargement des embeddings Word2Vec (GloVe)
    print("[INFO] Chargement des embeddings Word2Vec...")
    word_vectors = load('glove-twitter-100')

    # Construction de la matrice d'embeddings
    word_index = tokenizer.word_index
    num_words = min(len(word_index) + 1, args.max_features)
    embedding_matrix = np.zeros((num_words, args.embedding_dim))
    for word, i in word_index.items():
        if i >= args.max_features:
            continue
        embedding_matrix[i] = word_vectors[word] if word in word_vectors else np.random.normal(scale=0.6, size=(args.embedding_dim,))


    # Construction du modèle GRU
    model = Sequential([
        Embedding(
            input_dim=num_words,
            output_dim=args.embedding_dim,
            weights=[embedding_matrix],
            input_length=args.max_len,
            trainable=False
        ),
        Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
        Bidirectional(GRU(32, dropout=0.2, recurrent_dropout=0.2)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\n[INFO] Résumé du modèle :")
    model.summary()

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
    checkpoint_path = os.path.join(args.output_dir, "model_checkpoint_gru.keras")
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

    # Entraînement
    print("\n[INFO] Entraînement du modèle...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=64, callbacks=[early_stopping, checkpoint_callback]
    )
    end_time = time.time()

    # Temps d'entraînement
    training_time = end_time - start_time
    print(f"[INFO] Temps d'entraînement : {training_time:.2f} secondes")

    # Évaluation
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype('int32')
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Rapport de classification
    report = classification_report(y_test, y_pred)
    report_path = os.path.join(args.output_dir, "classification_report_GRU.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Sauvegarde du modèle et du tokenizer
    model_path = os.path.join(args.output_dir, "final_trained_gru_model.h5")
    tokenizer_path = os.path.join(args.output_dir, "tokenizer_gru.pkl")
    model.save(model_path)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)

    print("\n[INFO] Entraînement terminé.")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")
    print(f"Rapport de classification sauvegardé dans : {report_path}")
    print(f"Modèle sauvegardé dans : {model_path}")
    print(f"Tokenizer sauvegardé dans : {tokenizer_path}")

# --------------------------------------------------
# 3. Exécution du script
# --------------------------------------------------
if __name__ == "__main__":
    main()
