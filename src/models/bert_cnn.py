import tensorflow as tf
import numpy as np
import pandas as pd
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Dropout, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset
from azureml.core import Workspace
import mlflow
import mlflow.tensorflow
from src.models.common import plot_and_log_accuracy_loss


# Charger le workspace AzureML
try:
    ws = Workspace.from_config(path='/content/drive/MyDrive/AzureML/config.json')
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
    mlflow.set_experiment("SentimentAnalysis_BERT_CNN")
    print("AzureML Workspace chargé :", ws.name)
except Exception as e:
    print("Erreur de chargement du Workspace AzureML :", e)

# Charger le tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_and_sample_data(file_path, num_samples=10000):
    """
    Charger et échantillonner des tweets positifs et négatifs.
    """
    print("Chargement des données brutes...")
    data = pd.read_csv(file_path, encoding='latin1', header=None)
    data.columns = ["sentiment", "id", "date", "query", "user", "TweetText"]
    data['sentiment'] = data['sentiment'].replace(4, 1)  # Convertir 4 en 1 pour les positifs

    print("Échantillonnage de tweets positifs et négatifs...")
    positive_samples = data[data['sentiment'] == 1].sample(num_samples, random_state=42)
    negative_samples = data[data['sentiment'] == 0].sample(num_samples, random_state=42)
    sampled_data = pd.concat([positive_samples, negative_samples]).sample(frac=1, random_state=42).reset_index(drop=True)

    return sampled_data

def tokenize_and_encode(df, max_len=100):
    """
    Tokenise et encode les données avec BERT.
    """
    print("Tokenisation et encodage des données...")
    tokens = tokenizer(
        list(df['TweetText'].fillna("")),
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="tf"  # Spécifier TensorFlow
    )
    return tokens["input_ids"], tokens["attention_mask"]

class BertLayer(Layer):
    def __init__(self, **kwargs):
        super(BertLayer, self).__init__(**kwargs)
        self.bert = TFBertModel.from_pretrained('bert-base-uncased')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

def build_cnn_model(max_len):
    """
    Construire un modèle CNN pour les embeddings BERT.
    """
    print("Construction du modèle CNN...")
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    # Utiliser la couche personnalisée BERT
    bert_outputs = BertLayer()([input_ids, attention_mask])

    # CNN
    conv = Conv1D(128, kernel_size=3, activation="relu")(bert_outputs)
    pool = GlobalMaxPooling1D()(conv)
    dense = Dense(64, activation="relu")(pool)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation="sigmoid")(dropout)

    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-5), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_with_bert_cnn(file_path, num_samples=10000, max_len=100, epochs=5, batch_size=32):
    """
    Entraîne un modèle CNN avec des embeddings BERT et journalise les résultats avec MLflow.
    """
    # Étape 1 : Charger les données
    print("Chargement et échantillonnage des données...")
    data = load_and_sample_data(file_path, num_samples)

    # Étape 2 : Tokenisation et encodage
    print("Tokenisation et création des encodages...")
    input_ids, attention_mask = tokenize_and_encode(data, max_len)
    y = data['sentiment'].values

    # Étape 3 : Division des données
    print("Division des données...")
    X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
        input_ids.numpy(), attention_mask.numpy(), y, test_size=0.2, random_state=42, stratify=y
    )

    # Étape 4 : Créer un Dataset TensorFlow
    train_dataset = Dataset.from_tensor_slices(((X_train_ids, X_train_mask), y_train)).batch(batch_size)
    test_dataset = Dataset.from_tensor_slices(((X_test_ids, X_test_mask), y_test)).batch(batch_size)

    # Étape 5 : Construire le modèle CNN
    model = build_cnn_model(max_len)

    # Étape 6 : Activer l'autologging et entraîner
    print("Activation de l'autologging avec MLflow et début de l'entraînement...")
    mlflow.tensorflow.autolog()

    with mlflow.start_run(run_name="BERT_CNN"):
        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            verbose=1
        )
    # Générer et enregistrer les courbes dans MLflow
    plot_and_log_accuracy_loss(history)

        print("Entraînement terminé et enregistré avec MLflow.")
        return history

if __name__ == "__main__":
    train_with_bert_cnn(
        file_path="/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/training.csv",
        num_samples=10000,
        max_len=100,
        epochs=5,
        batch_size=32
    )
