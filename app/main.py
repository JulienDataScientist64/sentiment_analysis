from fastapi import FastAPI, Body
import tensorflow as tf
import numpy as np

# Initialisation de FastAPI
app = FastAPI()

# Charger le modèle avec TensorFlow
model = tf.keras.models.load_model("models/final_trained_light_LSTM.h5")


# Endpoint racine
@app.get("/")
def root():
    return {"message": "API is running"}


# Endpoint pour la prédiction
@app.post("/predict")
def predict(data: list = Body(...)):
    arr = np.array(data)
    preds = model.predict(arr).tolist()
    return {"predictions": preds}
