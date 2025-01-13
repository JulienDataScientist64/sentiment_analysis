from fastapi import FastAPI, Body, HTTPException
from tensorflow import keras
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List

# Définir un schéma Pydantic pour valider la requête
class TextsInput(BaseModel):
    texts: List[str]

app = FastAPI()

# Charger le modèle TensorFlow
MODEL_PATH = "models/lstm.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Gestion des erreurs de chargement du modèle et du tokenizer
try:
    model = keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

try:
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du tokenizer : {e}")

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(texts_input: TextsInput):
    try:
        # Récupérer les textes à partir de la requête
        texts = texts_input.texts

        # Prétraitement des séquences
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=50
        )  # Assure-toi que `maxlen` correspond à ton modèle

        # Prédiction
        predictions = model.predict(padded_sequences)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")
