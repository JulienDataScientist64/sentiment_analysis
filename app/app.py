from fastapi import FastAPI, Body, HTTPException
from tensorflow import keras
import pickle
import numpy as np
from pydantic import BaseModel
from typing import List

# Définir un schéma Pydantic pour valider la requête
class TextsInput(BaseModel):
    texts: List[str]

# Initialisation de l'application FastAPI
app = FastAPI()

# Chemins des modèles
MODEL_PATH = "models/lstm.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Charger le modèle et le tokenizer avec gestion d'erreur
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
        texts = texts_input.texts
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=50
        )
        predictions = model.predict(padded_sequences)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")
