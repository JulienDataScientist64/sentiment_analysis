import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model
import argparse
import os

# --------------------------------------------------
# 1. Chargement des données et Préparation
# --------------------------------------------------
def load_and_prepare_data(data_path, max_len):
    print(f"[INFO] Chargement des données depuis {data_path}...")
    balanced_data = pd.read_csv(data_path)[['target', 'TweetText']]
    train_data, temp_data = train_test_split(
        balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['target']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=temp_data['target']
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def preprocess_texts_with_bert(texts):
        tokens = tokenizer(
            list(texts),
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="tf"
        )
        return tokens['input_ids'], tokens['attention_mask']
    
    X_train_ids, X_train_mask = preprocess_texts_with_bert(train_data['TweetText'])
    X_val_ids, X_val_mask = preprocess_texts_with_bert(val_data['TweetText'])
    X_test_ids, X_test_mask = preprocess_texts_with_bert(test_data['TweetText'])
    y_train = tf.convert_to_tensor(train_data['target'].values, dtype=tf.int32)
    y_val = tf.convert_to_tensor(val_data['target'].values, dtype=tf.int32)
    y_test = tf.convert_to_tensor(test_data['target'].values, dtype=tf.int32)
    
    return (tf.convert_to_tensor(X_train_ids), tf.convert_to_tensor(X_train_mask), y_train), (tf.convert_to_tensor(X_val_ids), tf.convert_to_tensor(X_val_mask), y_val), (tf.convert_to_tensor(X_test_ids), tf.convert_to_tensor(X_test_mask), y_test)

# --------------------------------------------------
# 2. Création du modèle BERT + CNN
# --------------------------------------------------
def create_bert_model(max_len):
    input_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
    bert_model = TFBertModel.from_pretrained("bert-base-uncased")
    bert_outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    conv = Conv1D(128, kernel_size=3, activation="relu")(bert_outputs)
    pool = GlobalMaxPooling1D()(conv)
    dense = Dense(64, activation="relu")(pool)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation="sigmoid")(dropout)
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --------------------------------------------------
# 3. Entraînement et Évaluation
# --------------------------------------------------
def train_and_evaluate(model, train_data, val_data, test_data, max_len, output_dir):
    (X_train_ids, X_train_mask, y_train) = train_data
    (X_val_ids, X_val_mask, y_val) = val_data
    (X_test_ids, X_test_mask, y_test) = test_data
    start_time = time.time()
    history = model.fit(
        [X_train_ids, X_train_mask],
        y_train,
        validation_data=([X_val_ids, X_val_mask], y_val),
        epochs=5,
        batch_size=32,
        verbose=1
    )
    end_time = time.time()
    training_time = end_time - start_time
    print(f"[INFO] Temps d'entraînement: {training_time:.2f} secondes")
    y_pred_prob = model.predict([X_test_ids, X_test_mask])
    y_pred = (y_pred_prob > 0.5).astype('int32')
    y_test_cpu = y_test.numpy()
    test_accuracy = accuracy_score(y_test_cpu, y_pred)
    test_f1 = f1_score(y_test_cpu, y_pred)
    print(f"[INFO] Précision sur le test: {test_accuracy:.4f}")
    print(f"[INFO] Score F1 sur le test: {test_f1:.4f}")
    report = classification_report(y_test_cpu, y_pred)
    print("Classification Report:\n", report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

# --------------------------------------------------
# 4. Exécution du script principal
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement BERT + CNN")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers le fichier CSV des tweets.")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie pour les rapports et modèles.")
    parser.add_argument("--max_len", type=int, default=50, help="Longueur maximale des séquences.")
    args = parser.parse_args()
    train_data, val_data, test_data = load_and_prepare_data(args.data_path, args.max_len)
    model = create_bert_model(args.max_len)
    train_and_evaluate(model, train_data, val_data, test_data, args.max_len, args.output_dir)
