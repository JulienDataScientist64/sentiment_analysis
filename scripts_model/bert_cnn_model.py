import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

# --------------------------------------------------
# 1. Chargement et préparation des données
# --------------------------------------------------
def load_data(data_path, max_len, tokenizer):
    print(f"[INFO] Chargement des données depuis {data_path}...")
    data = pd.read_csv(data_path)[['target', 'TweetText']]

    # Division des données en train, validation, test
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['target'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['target'])

    def tokenize_texts(texts):
        return tokenizer(
            list(texts),
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="np"
        )

    X_train = tokenize_texts(train_data['TweetText'])
    X_val = tokenize_texts(val_data['TweetText'])
    X_test = tokenize_texts(test_data['TweetText'])

    y_train = train_data['target'].values
    y_val = val_data['target'].values
    y_test = test_data['target'].values

    return X_train, y_train, X_val, y_val, X_test, y_test

# --------------------------------------------------
# 2. Création du modèle BERT + Dense
# --------------------------------------------------
def create_model(max_len):
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")

    # Définir les inputs
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Sorties de BERT
    bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = bert_outputs.pooler_output

    # Couche Dense
    dropout = Dropout(0.3)(pooled_output)
    output = Dense(1, activation="sigmoid")(dropout)

    # Création du modèle
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

# --------------------------------------------------
# 3. Fonction principale
# --------------------------------------------------
def main(data_path, output_dir, max_len):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Charger les données
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(data_path, max_len, tokenizer)

    # Créer le modèle
    model = create_model(max_len)
    model.summary()

    # Entraîner le modèle
    print("\n[INFO] Début de l'entraînement...")
    model.fit(
        [X_train['input_ids'], X_train['attention_mask']],
        y_train,
        validation_data=([X_val['input_ids'], X_val['attention_mask']], y_val),
        epochs=3,
        batch_size=32,
        verbose=1
    )

    # Évaluer le modèle
    print("\n[INFO] Évaluation sur l'ensemble de test...")
    y_pred_prob = model.predict([X_test['input_ids'], X_test['attention_mask']])
    y_pred = (y_pred_prob > 0.5).astype(int)

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")

    # Sauvegarder le modèle et le rapport
    model_path = os.path.join(output_dir, "final_bert_dense_model.h5")
    report_path = os.path.join(output_dir, "classification_report_BERT_Dense.txt")

    model.save(model_path)
    print(f"[INFO] Modèle sauvegardé dans : {model_path}")

    report = classification_report(y_test, y_pred)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"[INFO] Rapport de classification sauvegardé dans : {report_path}")

# --------------------------------------------------
# 4. Exécution du script
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BERT + Dense Layer Sentiment Analysis")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers le fichier CSV des tweets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie pour les modèles et rapports.")
    parser.add_argument("--max_len", type=int, default=50, help="Longueur maximale des séquences.")
    args = parser.parse_args()

    main(args.data_path, args.output_dir, args.max_len)
