import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

import mlflow
import mlflow.sklearn

from google.colab import drive
drive.mount('/content/drive')

# --------------------------------------------------
# 1. Configuration MLflow (stockage Google Drive)
# --------------------------------------------------
mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")
mlflow.set_experiment("Sentiment_Analysis_Models")

# --------------------------------------------------
# 2. Chargement du DataFrame et préparation des données
# --------------------------------------------------
data_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/train_data.csv'
balanced_data = pd.read_csv(data_path)[['target', 'TweetText']]

train_data, temp_data = train_test_split(
    balanced_data,
    test_size=0.3,
    random_state=42,
    stratify=balanced_data['target']
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_data['target']
)

# --------------------------------------------------
# 3. Vectorisation TF-IDF
# --------------------------------------------------
max_features = 5000
vectorizer = TfidfVectorizer(max_features=max_features)

X_train_tfidf = vectorizer.fit_transform(train_data['TweetText'])
y_train = train_data['target']

X_val_tfidf = vectorizer.transform(val_data['TweetText'])
y_val = val_data['target']

X_test_tfidf = vectorizer.transform(test_data['TweetText'])
y_test = test_data['target']

# --------------------------------------------------
# 4. GridSearchCV pour la LogReg
# --------------------------------------------------
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [100, 500, 1000]
}

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    verbose=1,
    n_jobs=-1
)

# --------------------------------------------------
# 5. Entraînement + tracking MLflow
# --------------------------------------------------
with mlflow.start_run(run_name="TF-IDF_RegressionLogistique"):
    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("max_features", max_features)

    mlflow.sklearn.autolog()

    # GridSearch
    grid_search.fit(X_train_tfidf, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log des meilleurs hyperparamètres
    mlflow.log_params(best_params)

    # Validation
    y_val_pred = best_model.predict(X_val_tfidf)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("val_f1_score", val_f1)

    # Test
    y_test_pred = best_model.predict(X_test_tfidf)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1_score", test_f1)

    # Rapport de classification
    report = classification_report(y_test, y_test_pred)
    report_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/classification_report_LogReg.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # --------------------------------------------------
    # 6. Sauvegarde du modèle et du vectoriseur
    # --------------------------------------------------
    model_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/tfidf_logistic_regression_best.pkl'
    vect_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/tfidf_vectorizer.pkl'
    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vect_path)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(vect_path)

    print("\nValidation Accuracy :", val_accuracy)
    print("Validation F1 Score :", val_f1)
    print("\nTest Accuracy :", test_accuracy)
    print("Test F1 Score :", test_f1)
    print("\nClassification Report (Test):\n", report)

print("[INFO] Entraînement et évaluation terminés. Les logs sont dans 'file:/content/drive/MyDrive/mlruns'.")





import os
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import gensim.downloader as api

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Bidirectional, Dense, Dropout, Embedding, LSTM
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --------------------------------------------------
# 1. Vérification GPU & Activation XLA
# --------------------------------------------------
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print(f"[INFO] GPU détecté: {device_name}")
    tf.config.optimizer.set_jit(True)  # Active XLA pour accélérer un peu
else:
    print("[WARNING] Aucun GPU détecté, exécution sur CPU.")

# --------------------------------------------------
# 2. Configuration MLflow (stockage Google Drive)
# --------------------------------------------------
mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")
mlflow.set_experiment("Sentiment_Analysis_Models")

# --------------------------------------------------
# 3. Chargement du DataFrame
# --------------------------------------------------
data_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/train_data.csv'
balanced_data = pd.read_csv(data_path)[['target', 'TweetText']]

# --------------------------------------------------
# 4. Division train/val/test
# --------------------------------------------------
train_data, temp_data = train_test_split(
    balanced_data,
    test_size=0.3,
    random_state=42,
    stratify=balanced_data['target']
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_data['target']
)

# --------------------------------------------------
# 5. Tokenisation et préparation des séquences
# --------------------------------------------------
max_features = 10_000
max_len = 50

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_data['TweetText'])

def preprocess_texts(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len)

X_train = preprocess_texts(train_data['TweetText'])
X_val = preprocess_texts(val_data['TweetText'])
X_test = preprocess_texts(test_data['TweetText'])

y_train = train_data['target'].values
y_val = val_data['target'].values
y_test = test_data['target'].values

# --------------------------------------------------
# 6. Chargement des embeddings GloVe (glove-twitter-50)
#    => plus légers, dimension = 50
# --------------------------------------------------
embedding_dim = 50
word_vectors = api.load('glove-twitter-50')
print("[INFO] Embeddings GloVe (50-dim) chargés avec succès.")

word_index = tokenizer.word_index
num_words = min(len(word_index) + 1, max_features)

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_features:
        continue
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# --------------------------------------------------
# 7. Construction d'un modèle LSTM plus léger
#    - Une seule couche LSTM, 32 unités
#    - Bidirectionnel
# --------------------------------------------------
model = Sequential([
    Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_shape=(max_len,),
        trainable=False
    ),
    Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------------------------------
# 8. Définition des callbacks
# --------------------------------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint_lstm_light.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_weights_only=False,
    verbose=1
)

# --------------------------------------------------
# 9. Entraînement + Tracking MLflow
# --------------------------------------------------
with mlflow.start_run(run_name='Light_GloVe_LSTM', nested=True):
    mlflow.log_param('model', 'Light_Bidirectional_LSTM')
    mlflow.log_param('embedding_dim', embedding_dim)
    mlflow.log_param('max_features', max_features)
    mlflow.log_param('max_len', max_len)

    # Sauvegarder le tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    mlflow.log_artifact('tokenizer.pkl')

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
        verbose=1,
        callbacks=[early_stopping, checkpoint_callback]
    )
    end_time = time.time()

    # Temps d'entraînement
    training_time = end_time - start_time
    mlflow.log_metric('training_time', training_time)

    # Évaluation
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype('int32')

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)

    # Rapport de classification
    report = classification_report(y_test, y_pred)
    mlflow.log_text(report, 'classification_report_Light_LSTM.txt')

    # Sauvegarde du modèle final
    model.save('final_trained_light_LSTM.h5')
    mlflow.log_artifact('final_trained_light_LSTM.h5')

print("[INFO] Entraînement terminé. Logs enregistrés sur 'file:/content/drive/MyDrive/mlruns'.")




##########################################################
# Code optimisé pour la classification de sentiments
# via embeddings GloVe + GRU bidirectionnel, suivi MLflow
##########################################################

import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

import gensim.downloader as api
import mlflow
import mlflow.tensorflow

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Bidirectional, Dense, Dropout, Embedding,
                                     GRU)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --------------------------------------------------
# 1. Vérifier la dispo du GPU et activer XLA (optionnel)
# --------------------------------------------------
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print("GPU détecté :", device_name)
    # Optionnel : activer XLA pour booster les perfs
    tf.config.optimizer.set_jit(True)  # Active l'optimisation XLA
else:
    print("Aucun GPU détecté, exécution sur CPU.")

# --------------------------------------------------
# 2. Configuration de MLflow : stockage dans Google Drive
# --------------------------------------------------
mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")
mlflow.set_experiment("Sentiment_Analysis_Models")

# --------------------------------------------------
# 3. Chargement du DataFrame
# --------------------------------------------------
data_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/train_data.csv'
balanced_data = pd.read_csv(data_path)
balanced_data = balanced_data[['target', 'TweetText']]

# --------------------------------------------------
# 4. Préparation des données : train / validation / test
# --------------------------------------------------
train_data, temp_data = train_test_split(
    balanced_data,
    test_size=0.3,
    random_state=42,
    stratify=balanced_data['target']
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_data['target']
)

# --------------------------------------------------
# 5. Paramètres de tokenisation et prétraitement
# --------------------------------------------------
max_features = 10000  # Nombre max de mots à considérer
max_len = 50          # Longueur max des séquences

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_data['TweetText'])

def preprocess_texts(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_len)

X_train = preprocess_texts(train_data['TweetText'])
X_val = preprocess_texts(val_data['TweetText'])
X_test = preprocess_texts(test_data['TweetText'])

y_train = train_data['target'].values
y_val = val_data['target'].values
y_test = test_data['target'].values

# --------------------------------------------------
# 6. Chargement des embeddings GloVe
# --------------------------------------------------
embedding_dim = 100
print("Chargement des embeddings GloVe...")
word_vectors = api.load('glove-twitter-100')  # Embeddings légers
print("Embeddings GloVe chargés avec succès.")

# --------------------------------------------------
# 7. Construction de la matrice d'embeddings
# --------------------------------------------------
word_index = tokenizer.word_index
num_words = min(len(word_index) + 1, max_features)
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i >= max_features:
        continue
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# --------------------------------------------------
# 8. Construction du modèle GRU Bidirectionnel
# --------------------------------------------------
model = Sequential([
    Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_shape=(max_len,),
        trainable=False  # Mettre True si on veut affiner l'embedding
    ),
    Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)),
    Bidirectional(GRU(32, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Résumé du modèle :")
model.summary()

# --------------------------------------------------
# 9. Callbacks Keras (EarlyStopping, Checkpoint)
# --------------------------------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_weights_only=False,  # on veut sauvegarder la structure+poids
    verbose=1
)

# --------------------------------------------------
# 10. Entraînement + tracking MLflow
# --------------------------------------------------
with mlflow.start_run(run_name='Optimized_Word2Vec_GRU', nested=True):
    mlflow.log_param('model', 'Bidirectional_GRU')
    mlflow.log_param('embedding_dim', embedding_dim)
    mlflow.log_param('max_features', max_features)
    mlflow.log_param('max_len', max_len)

    # Sauvegarder le tokenizer comme artefact
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    mlflow.log_artifact('tokenizer.pkl')

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,  # batch size possible à ajuster selon la RAM GPU
        verbose=1,
        callbacks=[early_stopping, checkpoint_callback]
    )
    end_time = time.time()

    # Logguer le temps d'entraînement
    training_time = end_time - start_time
    mlflow.log_metric('training_time', training_time)

    # --------------------------------------------------
    # Évaluation
    # --------------------------------------------------
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype('int32')
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)

    # Enregistrer le rapport de classification dans MLflow
    report = classification_report(y_test, y_pred)
    mlflow.log_text(report, 'classification_report_Optimized_GRU.txt')
    print("\nClassification Report:\n", report)

    # --------------------------------------------------
    # Sauvegarde finale du meilleur modèle pour déploiement
    # --------------------------------------------------
    # ModelCheckpoint a déjà sauvegardé le meilleur modèle,
    # mais on peut aussi sauvegarder la version finale
    model.save('final_trained_model.h5')
    mlflow.log_artifact('final_trained_model.h5')

print("Entraînement et évaluation terminés. Les logs sont dans 'file:/content/drive/MyDrive/mlruns'.")




import os
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow
import mlflow.tensorflow
import gensim.downloader as api

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --------------------------------------------------
# 1. Vérification GPU & Activation XLA (optionnel)
# --------------------------------------------------
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print(f"[INFO] GPU détecté: {device_name}")
    tf.config.optimizer.set_jit(True)  # Active XLA (peut accélérer un peu)
else:
    print("[WARNING] Aucun GPU détecté, exécution sur CPU.")

# --------------------------------------------------
# 2. Configuration MLflow (Google Drive)
# --------------------------------------------------
mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")
mlflow.set_experiment("Sentiment_Analysis_Models")

# --------------------------------------------------
# 3. Chargement du DataFrame
#     Ex.: 200k tweets (100k neg, 100k pos)
# --------------------------------------------------
data_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/train_data.csv'
balanced_data = pd.read_csv(data_path)[['target', 'TweetText']]

# --------------------------------------------------
# 4. Division train/val/test (70%/15%/15%)
# --------------------------------------------------
train_data, temp_data = train_test_split(
    balanced_data,
    test_size=0.3,
    random_state=42,
    stratify=balanced_data['target']
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=42,
    stratify=temp_data['target']
)

# --------------------------------------------------
# 5. Tokenisation et préparation des séquences
# --------------------------------------------------
max_features = 10_000  # Ajuste si besoin (vocab)
max_len = 50

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_data['TweetText'])

def preprocess_texts(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len)

X_train = preprocess_texts(train_data['TweetText'])
X_val = preprocess_texts(val_data['TweetText'])
X_test = preprocess_texts(test_data['TweetText'])

y_train = train_data['target'].values
y_val = val_data['target'].values
y_test = test_data['target'].values

# --------------------------------------------------
# 6. Chargement embeddings GloVe (twitter-50)
#    -> 50 dimensions (plus léger)
# --------------------------------------------------
embedding_dim = 50
word_vectors = api.load('glove-twitter-50')
print("[INFO] Embeddings GloVe (50-dim) chargés avec succès.")

word_index = tokenizer.word_index
num_words = min(len(word_index) + 1, max_features)

embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i >= max_features:
        continue
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]
    else:
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))

# --------------------------------------------------
# 7. Modèle GRU allégé
#    - Une seule couche GRU (64)
#    - Dense réduit (32)
# --------------------------------------------------
model = Sequential([
    Embedding(
        input_dim=num_words,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_shape=(max_len,),
        trainable=False  # Ne pas affiner l'embedding, + léger
    ),
    Bidirectional(GRU(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --------------------------------------------------
# 8. Callbacks
# --------------------------------------------------
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

checkpoint_callback = ModelCheckpoint(
    filepath='model_checkpoint_light.keras',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    save_weights_only=False,
    verbose=1
)

# --------------------------------------------------
# 9. Entraînement + Tracking MLflow
# --------------------------------------------------
with mlflow.start_run(run_name='Light_GloVe_GRU', nested=True):
    mlflow.log_param('model', 'Bidirectional_GRU_light')
    mlflow.log_param('embedding', 'glove-twitter-50')
    mlflow.log_param('embedding_dim', embedding_dim)
    mlflow.log_param('max_features', max_features)
    mlflow.log_param('max_len', max_len)

    # Sauvegarde du tokenizer
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    mlflow.log_artifact('tokenizer.pkl')

    print("[INFO] Début de l'entraînement ...")
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,  # 128 possible si tu as plus de mémoire
        verbose=1,
        callbacks=[early_stopping, checkpoint_callback]
    )
    end_time = time.time()

    training_time = end_time - start_time
    mlflow.log_metric('training_time', training_time)

    print("[INFO] Évaluation sur l'ensemble de test ...")
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype('int32')

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)

    report = classification_report(y_test, y_pred)
    mlflow.log_text(report, 'classification_report_Light_GloVe_GRU.txt')

    # Sauvegarde du modèle final
    model.save('final_light_GloVe_GRU.h5')
    mlflow.log_artifact('final_light_GloVe_GRU.h5')

    print(f"[INFO] Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
    print("[INFO] Rapport de classification sauvegardé dans MLflow.")

print("[INFO] Modèle entraîné. Logs dans 'file:/content/drive/MyDrive/mlruns'.")



import time
import numpy as np
import pandas as pd
import tensorflow as tf
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.models import Model


mlflow.set_tracking_uri("file:/content/drive/MyDrive/mlruns")
mlflow.set_experiment("Sentiment_Analysis_Models")
# --------------------------------------------------
# 1. Chargement des données
# --------------------------------------------------
data_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/train_data.csv'
balanced_data = pd.read_csv(data_path)

# On garde uniquement les colonnes utiles
balanced_data = balanced_data[['target', 'TweetText']]

# --------------------------------------------------
# 2. Division des données en train, validation, test
# --------------------------------------------------
train_data, temp_data = train_test_split(
    balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['target']
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, random_state=42, stratify=temp_data['target']
)

# --------------------------------------------------
# 3. Préparer le tokenizer BERT
# --------------------------------------------------
max_len = 50
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

# --------------------------------------------------
# 4. Prétraitement des données
# --------------------------------------------------
X_train_ids, X_train_mask = preprocess_texts_with_bert(train_data['TweetText'])
X_val_ids, X_val_mask = preprocess_texts_with_bert(val_data['TweetText'])
X_test_ids, X_test_mask = preprocess_texts_with_bert(test_data['TweetText'])

# --------------------------------------------------
# 5. Conversion explicite en tf.Tensor
# --------------------------------------------------
X_train_ids = tf.convert_to_tensor(X_train_ids, dtype=tf.int32)
X_train_mask = tf.convert_to_tensor(X_train_mask, dtype=tf.int32)
X_val_ids = tf.convert_to_tensor(X_val_ids, dtype=tf.int32)
X_val_mask = tf.convert_to_tensor(X_val_mask, dtype=tf.int32)
X_test_ids = tf.convert_to_tensor(X_test_ids, dtype=tf.int32)
X_test_mask = tf.convert_to_tensor(X_test_mask, dtype=tf.int32)

# Extraction des labels et conversion en tf.Tensor
y_train = tf.convert_to_tensor(train_data['target'].values, dtype=tf.int32)
y_val = tf.convert_to_tensor(val_data['target'].values, dtype=tf.int32)
y_test = tf.convert_to_tensor(test_data['target'].values, dtype=tf.int32)

# --------------------------------------------------
# 6. Définition du modèle BERT + CNN
# --------------------------------------------------
def create_bert_model(max_len):
    """
    Crée un modèle Keras en utilisant une couche personnalisée
    qui encapsule TFBertModel.
    """
    # 1) Inputs du modèle
    input_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="input_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="attention_mask"
    )

    # 2) Définition de la couche BERT personnalisée
    class BertLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(BertLayer, self).__init__(**kwargs)
            # On charge le modèle pré-entraîné de Hugging Face
            self.bert = TFBertModel.from_pretrained("bert-base-uncased")

        def call(self, inputs):
            input_ids, attention_mask = inputs
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Renvoie uniquement la dernière couche cachée
            return outputs.last_hidden_state

    # 3) Application de la couche BertLayer
    bert_outputs = BertLayer()([input_ids, attention_mask])

    # 4) Ajout des couches CNN
    conv = Conv1D(128, kernel_size=3, activation="relu")(bert_outputs)
    pool = GlobalMaxPooling1D()(conv)
    dense = Dense(64, activation="relu")(pool)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation="sigmoid")(dropout)

    # 5) Construction et compilation du modèle
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model

# --------------------------------------------------
# 7. Instanciation du modèle
# --------------------------------------------------
model = create_bert_model(max_len)

# --------------------------------------------------
# 8. Entraînement & suivi MLflow
# --------------------------------------------------
mlflow.set_experiment("Sentiment_Analysis_Models")
with mlflow.start_run(run_name='Optimized_BERT_CNN', nested=True):
    # Log des hyperparamètres
    mlflow.log_param('model', 'BERT_CNN')
    mlflow.log_param('max_len', max_len)

    # Sauvegarde du tokenizer
    tokenizer.save_pretrained('bert_tokenizer')
    mlflow.log_artifacts('bert_tokenizer', artifact_path='tokenizer')

    # Entraînement
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

    # Log du temps d'entraînement
    training_time = end_time - start_time
    mlflow.log_metric('training_time', training_time)

    # Évaluation
    y_pred_prob = model.predict([X_test_ids, X_test_mask])  # Numpy array
    y_pred = (y_pred_prob > 0.5).astype('int32')            # Numpy array (CPU)

# Convertir y_test en Numpy array (CPU)
    y_test_cpu = y_test.numpy()

    test_accuracy = accuracy_score(y_test_cpu, y_pred)
    test_f1 = f1_score(y_test_cpu, y_pred)

    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)

# Rapport de classification
    report = classification_report(y_test_cpu, y_pred)
    print("Classification Report:\n", report)

print("Training and evaluation complete.")
