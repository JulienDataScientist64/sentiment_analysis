import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# Configurer le logger pour Application Insights
logger = logging.getLogger(__name__)
logger.addHandler(AzureLogHandler(connection_string="InstrumentationKey=44a30365-a80d-4a88-9886-eb58a8c02712"))

# Charger le modèle et le tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("/content/drive/MyDrive/sentiment_analysis/models/final_trained_light_LSTM.h5")
    with open("/content/drive/MyDrive/sentiment_analysis/models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

# Charger le modèle
model, tokenizer = load_model()

# Interface utilisateur Streamlit
st.title("Sentiment Analysis App")
st.subheader("Entrez un texte à analyser :")

user_input = st.text_area("Texte", "")

if st.button("Prédire"):
    if user_input:
        # Prétraitement du texte
        seq = tokenizer.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=50)

        # Prédiction
        prediction = model.predict(padded_seq)
        sentiment = "Positif" if prediction[0] > 0.5 else "Négatif"
        confidence = round(float(prediction[0][0]) * 100, 2)

        st.write(f"**Résultat : {sentiment} ({confidence}%)**")

        # Demander une validation
        feedback = st.radio("La prédiction est-elle correcte ?", ("Oui", "Non"))

        if feedback == "Non":
            logger.warning("Prédiction incorrecte signalée.")
            st.write("⚠️ Merci pour votre retour ! La prédiction incorrecte a été signalée.")

    else:
        st.write("⚠️ Veuillez entrer un texte.")
