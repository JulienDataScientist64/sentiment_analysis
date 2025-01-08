from google.colab import drive
drive.mount('/content/drive')

# Aller dans le répertoire sentiment_analysis sur Drive
%cd /content/drive/MyDrive/sentiment_analysis


# Création du fichier README.md
with open("README.md", "w") as f:
    f.write("# sentiment_analysis\n\nAnalyse de sentiments sur Twitter.\n")


# Création du fichier .gitignore
with open(".gitignore", "w") as gitignore:
    gitignore.write("""\
# Ignorer les fichiers CSV et volumineux
*.csv
*.pkl
*.npz

# Ignorer les checkpoints Jupyter Notebook
.ipynb_checkpoints/

# Ignorer les caches Python
__pycache__/
*.py[cod]

# Ignorer les fichiers temporaires ou de sauvegarde
*.tmp
*.log
*.bak
*.swp
~*

# Ignorer les fichiers système
.DS_Store

# Ignorer le dossier data/
data/
""")


# Initialiser le dépôt Git
!git init


# Ajouter les fichiers au commit
!git add README.md .gitignore

# Faire le commit
!git commit -m "First commit: Ajout du README et du .gitignore"


# Configurer le nom d'utilisateur et l'adresse email
!git config --global user.name "JulienDataScientist64"
!git config --global user.email "jcantalapiedra1@gmail.com"


# Créer le dossier .ssh
!mkdir -p ~/.ssh

# Copier les clés SSH depuis Drive
!cp /content/drive/MyDrive/ssh_keys/id_rsa ~/.ssh/id_rsa
!cp /content/drive/MyDrive/ssh_keys/id_rsa.pub ~/.ssh/id_rsa.pub

# Définir les permissions
!chmod 600 ~/.ssh/id_rsa

# Ajouter GitHub aux hôtes connus
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# Vérifier la connexion SSH avec GitHub
!ssh -T git@github.com


# Vérifie si un remote 'origin' existe déjà
!git remote -v

# Si un remote 'origin' existe déjà et est incorrect, supprime-le
!git remote remove origin

# Ajouter le remote correct
!git remote add origin git@github.com:JulienDataScientist64/sentiment_analysis.git

!git branch -M main


!git push -u origin main



# Ajouter tous les fichiers dans le répertoire notebook
!git add notebook/*

# Faire un commit avec un message
!git commit -m "Ajout des notebooks d'analyse de sentiments"

# Pousser le commit vers GitHub
!git push


# Afficher les modifications
!git status


import pandas as pd
import warnings
from IPython.display import display, Markdown
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration des options de pandas pour l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:,.2f}'.format
pd.options.mode.chained_assignment = None

# Gestion des avertissements
warnings.filterwarnings('ignore', category=DeprecationWarning)

def display_info(key, value):
    """ Affiche les informations dans un format markdown. """
    display(Markdown(f"**{key}:** {value}"))

# Fonction de séparation pour affichage clair
def display_separator():
    """ Affiche un séparateur markdown. """
    display(Markdown("-" * 80))

# Fonction pour afficher le docstring
def display_docstring(func):
    display(Markdown(f"**Documentation for {func.__name__}:**\n\n{func.__doc__}"))


from google.colab import drive
drive.mount('/content/drive')

# Generer une clef SSh pour GitHub
!ssh-keygen -t rsa -b 4096 -C "jcantalapiedra1@gmail.com" -f ~/.ssh/id_rsa -N ""

# Afficher ka clef SSh pour la copier dans Git
!cat ~/.ssh/id_rsa.pub

# Créer le dossier sur mon drive avec copie des clefs
!mkdir -p /content/drive/MyDrive/ssh_keys  # Créer le dossier ssh_keys sur Google Drive
!cp ~/.ssh/id_rsa /content/drive/MyDrive/ssh_keys/id_rsa  # Copier la clé privée
!cp ~/.ssh/id_rsa.pub /content/drive/MyDrive/ssh_keys/id_rsa.pub  # Copier la clé publique

# A chaque redemarrage de session colab
!mkdir -p ~/.ssh  # Créer le répertoire SSH
!cp /content/drive/MyDrive/ssh_keys/id_rsa ~/.ssh/id_rsa  # Copier la clé privée
!cp /content/drive/MyDrive/ssh_keys/id_rsa.pub ~/.ssh/id_rsa.pub  # Copier la clé publique
!chmod 600 ~/.ssh/id_rsa  # Appliquer les permissions correctes à la clé privée
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts  # Ajouter GitHub aux hôtes connus
!ssh -T git@github.com  # Tester la connexion SSH

# Définition du chemin des données
file_path =  '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/training.csv'
data = pd.read_csv(file_path, encoding='latin1')
data.head()

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]
data.columns = DATASET_COLUMNS
data.head()  # Affiche les premières lignes des données
data.info()  # Affiche la structure des données

# Ajouter le dossier src au path Python
import sys
sys.path.append('/content/drive/MyDrive/sentiment_analysis/src/')

# Importer la classe SentimentEDA
from sentiment_eda import SentimentEDA

# Créer une instance et exécuter le script
eda = SentimentEDA(file_path='/content/drive/MyDrive/sentiment_analysis/training.csv')
eda.load_data()
eda.perform_eda()


# Lister les fichiers et dossiers
!ls -R /content/drive/MyDrive/sentiment_analysis


# Aller dans le dossier sentiment_analysis
%cd /content/drive/MyDrive/sentiment_analysis

# Ajouter le fichier sentiment_eda.py
!git add src/sentiment_eda.py

# Ajouter le fichier sentiment_eda.py
!git add src/sentiment_eda.py

# Faire un commit avec un message clair
!git commit -m "Ajout du script sentiment_eda.py"

# Pousser les modifications
!git push

# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Configurer SSH
!mkdir -p ~/.ssh
!cp /content/drive/MyDrive/ssh_keys/id_rsa ~/.ssh/id_rsa
!cp /content/drive/MyDrive/ssh_keys/id_rsa.pub ~/.ssh/id_rsa.pub
!chmod 600 ~/.ssh/id_rsa
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
!ssh -T git@github.com

# Configurer Git (Nom et email)
!git config --global user.name "JulienDataScientist64"
!git config --global user.email "jcantalapiedra1@gmail.com"

# Naviguer dans le répertoire de ton projet (cloné ou existant dans Colab)
%cd /content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning

# Synchroniser avec les changements distants
!git pull origin main --allow-unrelated-histories

# Vérifier l'état des modifications
!git status

# Ajouter toutes les modifications
!git add .

# Créer un commit avec une description claire
!git commit -m "Description des changements du jour"

# Pousser les modifications vers GitHub
!git push origin main


#Procédure complète pour un nouveau projet dans Colab#
from google.colab import drive
import os

# Étape 1 : Monter Google Drive
drive.mount('/content/drive')

# Étape 2 : Configurer SSH
!mkdir -p ~/.ssh
!cp /content/drive/MyDrive/ssh_keys/id_rsa ~/.ssh/id_rsa
!cp /content/drive/MyDrive/ssh_keys/id_rsa.pub ~/.ssh/id_rsa.pub
!chmod 600 ~/.ssh/id_rsa
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
!ssh -T git@github.com

# Étape 3 : Configurer Git (Nom et email)
!git config --global user.name "JulienDataScientist64"
!git config --global user.email "jcantalapiedra1@gmail.com"

# Étape 4 : Créer un répertoire pour le projet
new_project_path = '/content/drive/MyDrive/mon_nouveau_projet'
os.makedirs(new_project_path, exist_ok=True)
%cd {new_project_path}

# Étape 5 : Créer un fichier README.md
with open("README.md", "w") as f:
    f.write("# Mon Nouveau Projet\n\nDescription de mon projet.")

# Étape 6 : Initialiser Git dans le répertoire
!git init

# Étape 7 : Créer un fichier .gitignore
with open(".gitignore", "w") as gitignore:
    gitignore.write("""
training.csv
*.csv
__pycache__/
*.pkl
.DS_Store
config.json

# Ignorer les fichiers PNG
*.png

# Ignorer les fichiers temporaires ou de sauvegarde
*.tmp
*.log
*.bak
*.swp
~*
->>rajouter pour ignorer data set et touts les fichier volumineux sans importance
# Ignorer les sorties de Jupyter Notebook
.ipynb_checkpoints/
""")

# Étape 8 : Ajouter et commiter les fichiers
!git add .
!git commit -m "Initial commit: ajout du README et configuration initiale"

# Étape 9 : Lier le dépôt GitHub (remplace par le lien de ton dépôt GitHub)
!git remote add origin git@github.com:JulienDataScientist64/mon_nouveau_projet.git

# Étape 10 : Pousser les modifications vers GitHub
!git push -u origin main


import pandas as pd

# Chemin vers le fichier d'entrée
FILE_PATH = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/training.csv'

# Nom des colonnes du dataset
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "TweetText"]

# Chargement du dataset avec le bon encodage
data = pd.read_csv(FILE_PATH, encoding='latin1', names=DATASET_COLUMNS)

# Suppression des doublons pour éviter les répétitions
data.drop_duplicates(subset="TweetText", inplace=True)

# Conversion des labels : remplacer le label 4 par 1 (sentiment positif)
data['target'] = data['target'].replace(4, 1)

# **Étape 1 : Création du dataset de test équilibré**
# Échantillonnage de 2500 tweets négatifs et 2500 tweets positifs
test_negatives = data[data['target'] == 0].sample(2500, random_state=42)
test_positives = data[data['target'] == 1].sample(2500, random_state=42)

# Concaténation et mélange aléatoire des échantillons pour le dataset de test
hidden_data = pd.concat([test_negatives, test_positives]).sample(frac=1, random_state=42).reset_index(drop=True)

# Sauvegarde du dataset de test
hidden_data_PATH = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/test_data.csv'
hidden_data.to_csv(hidden_data_PATH, index=False)

# Suppression des tweets de test du dataset principal
data = data[~data['TweetText'].isin(hidden_data['TweetText'])]

# **Étape 2 : Création d'un dataset équilibré pour l'entraînement**
# Échantillonnage équilibré de 100,000 tweets pour chaque classe
negatives = data[data['target'] == 0].sample(100000, random_state=42)
positives = data[data['target'] == 1].sample(100000, random_state=42)

# Concaténation et mélange aléatoire des échantillons pour l'entraînement
balanced_data = pd.concat([negatives, positives]).sample(frac=1, random_state=42).reset_index(drop=True)

# Sauvegarde du dataset équilibré d'entraînement
TRAIN_DATA_PATH = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/train_data.csv'
balanced_data.to_csv(TRAIN_DATA_PATH, index=False)

# **Vérifications des données**
# Taille et répartition des classes dans le dataset hidden_data
print("Taille des données hidden_data :", hidden_data.shape)
print("Répartition des classes dans le dataset hidden_data :\n", hidden_data['target'].value_counts())

# Taille et répartition des classes dans le dataset d'entraînement
print("\nTaille des données d'entraînement :", balanced_data.shape)
print("Répartition des classes dans le dataset d'entraînement :\n", balanced_data['target'].value_counts())

# Aperçu des données
print("\nAperçu des données hidden_data :")
print(hidden_data.head())
print("\nColonnes du DataFrame d'entraînement :", balanced_data.columns)
print("\nAperçu des données d'entraînement :")
print(balanced_data.head())


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


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel
import mlflow
import mlflow.tensorflow
from azureml.core import Workspace
import tensorflow as tf

# Charger le Workspace Azure ML
ws = Workspace.from_config(path='/content/drive/MyDrive/config.json')
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
mlflow.set_experiment('Sentiment_Analysis_Models')

# Activer l'autologging
mlflow.tensorflow.autolog()

# Charger le DataFrame
data_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/train_data.csv'
balanced_data = pd.read_csv(data_path)
balanced_data = balanced_data[['target', 'TweetText']]

# Diviser les données en ensembles d'entraînement, validation et test
train_data, temp_data = train_test_split(
    balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['target']
)
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, random_state=42, stratify=temp_data['target']
)

# Préparer le tokenizer BERT
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

# Prétraitement des données
X_train_ids, X_train_mask = preprocess_texts_with_bert(train_data['TweetText'])
X_val_ids, X_val_mask = preprocess_texts_with_bert(val_data['TweetText'])
X_test_ids, X_test_mask = preprocess_texts_with_bert(test_data['TweetText'])

# Convertir en TensorFlow Tensors
X_train_ids = tf.convert_to_tensor(X_train_ids, dtype=tf.int32)
X_train_mask = tf.convert_to_tensor(X_train_mask, dtype=tf.int32)
X_val_ids = tf.convert_to_tensor(X_val_ids, dtype=tf.int32)
X_val_mask = tf.convert_to_tensor(X_val_mask, dtype=tf.int32)
X_test_ids = tf.convert_to_tensor(X_test_ids, dtype=tf.int32)
X_test_mask = tf.convert_to_tensor(X_test_mask, dtype=tf.int32)

y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

def create_bert_model(max_len):
    # Define model inputs
    input_ids = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="input_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(max_len,), dtype=tf.int32, name="attention_mask"
    )

    # Define a custom layer that wraps TFBertModel
    class BertLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(BertLayer, self).__init__(**kwargs)
            self.bert = TFBertModel.from_pretrained("bert-base-uncased")

        def call(self, inputs):
            input_ids, attention_mask = inputs
            outputs = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )
            return outputs.last_hidden_state

    # Use the custom BertLayer
    bert_outputs = BertLayer()([input_ids, attention_mask])

    # Add CNN layers
    conv = Conv1D(128, kernel_size=3, activation="relu")(bert_outputs)
    pool = GlobalMaxPooling1D()(conv)
    dense = Dense(64, activation="relu")(pool)
    dropout = Dropout(0.5)(dense)
    output = Dense(1, activation="sigmoid")(dropout)

    # Build and compile the model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model

model = create_bert_model(max_len)

# Entraîner et logguer le modèle avec MLflow
with mlflow.start_run(run_name='Optimized_BERT_CNN', nested=True):
    # Logguer les paramètres clés du modèle
    mlflow.log_param('model', 'BERT_CNN')
    mlflow.log_param('max_len', max_len)

    # Sauvegarder le tokenizer comme artefact
    tokenizer.save_pretrained('bert_tokenizer')
    mlflow.log_artifacts('bert_tokenizer', artifact_path='tokenizer')

    # Entraîner le modèle
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

    # Logguer le temps d'entraînement
    training_time = end_time - start_time
    mlflow.log_metric('training_time', training_time)

    # Évaluer le modèle
    y_pred_prob = model.predict([X_test_ids, X_test_mask])
    y_pred = (y_pred_prob > 0.5).astype('int32')
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    # Logguer les métriques finales
    mlflow.log_metric('test_accuracy', test_accuracy)
    mlflow.log_metric('test_f1_score', test_f1)

    # Logguer le rapport de classification
    report = classification_report(y_test, y_pred)
    mlflow.log_text(report, 'classification_report_Optimized_BERT.txt')

print("Training and evaluation complete.")


!pip install mlflow --quiet

import os
import shutil
import pprint

from random import random, randint
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts
from sklearn.ensemble import RandomForestRegressor
from mlflow.tracking import MlflowClient
import warnings

!pip install --upgrade mlflow


from google.colab import drive
drive.mount('/content/drive')

from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="file:/content/drive/MyDrive/mlruns")
experiments = client.search_experiments()
for e in experiments:
    print(f"[ExpID={e.experiment_id}] name={e.name}, artifact={e.artifact_location}")




from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="file:/content/drive/MyDrive/mlruns")

runs = client.search_runs(
    experiment_ids=["165120806529865697"],
    order_by=["attributes.start_time DESC"]
)

for r in runs:
    print(r.info.run_id, r.data.metrics, r.data.params)



!pip install mlflow pyngrok --quiet

from pyngrok import ngrok

# (Optionnel) tuer d’éventuels anciens tunnels
ngrok.kill()

# Lancer MLflow en tâche de fond
get_ipython().system_raw('mlflow ui --backend-store-uri="file:/content/drive/MyDrive/mlruns" --port 5000 &')

# Ouvrir le tunnel ngrok
public_url = ngrok.connect(5000)
print("MLflow Tracking UI disponible à l'adresse :", public_url.public_url)


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




import pandas as pd

# Charger le fichier test_data.csv
test_data_path = '/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/test_data.csv'
test_data = pd.read_csv(test_data_path)

# Afficher les premières lignes pour vérifier
display(test_data.head())



!pip install pipreqs



# Aller dans le répertoire sentiment_analysis sur Drive
%cd /content/drive/MyDrive/sentiment_analysis

# Installer pip-tools
!pip install pip-tools

# Générer le fichier requirements.txt
!pip-compile


# Naviguer dans le dossier sentiment_analysis
%cd /content/drive/MyDrive/sentiment_analysis

# Convertir le notebook en script Python
!jupyter nbconvert --to script /content/drive/MyDrive/sentiment_analysis/notebook/analyse\ de\ sentiments.ipynb


# Créer un dossier temporaire
!mkdir /content/drive/MyDrive/sentiment_analysis/temp_notebook

# Copier le fichier Python converti dans le dossier temporaire
!cp /content/drive/MyDrive/sentiment_analysis/notebook/analyse\ de\ sentiments.py /content/drive/MyDrive/sentiment_analysis/temp_notebook/


# Naviguer dans le dossier temporaire
%cd /content/drive/MyDrive/sentiment_analysis/temp_notebook

# Générer le fichier requirements.txt
!pipreqs . --force


# Déplacer requirements.txt vers le dossier sentiment_analysis
!mv /content/drive/MyDrive/sentiment_analysis/temp_notebook/requirements.txt /content/drive/MyDrive/sentiment_analysis/


!cat /content/drive/MyDrive/sentiment_analysis/requirements.txt
