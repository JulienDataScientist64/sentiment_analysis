import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# --------------------------------------------------------------------
# 1. Configurer le logger pour Application Insights
# --------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger.addHandler(
    AzureLogHandler(
        connection_string="InstrumentationKey=44a30365-a80d-4a88-9886-eb58a8c02712"
    )
)

# --------------------------------------------------------------------
# 2. Charger le modèle et le tokenizer
# --------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "/content/drive/MyDrive/sentiment_analysis/models/final_trained_light_LSTM.h5"
    )
    with open("/content/drive/MyDrive/sentiment_analysis/models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# --------------------------------------------------------------------
# 3. Interface utilisateur
# --------------------------------------------------------------------
st.title("Sentiment Analysis App")
st.subheader("Entrez un texte à analyser :")

# Champ de saisie pour le texte utilisateur
user_text = st.text_area("Texte", key="user_text_area")

# Bouton pour effectuer la prédiction
if st.button("Prédire"):
    if not user_text.strip():
        st.warning("Veuillez entrer un texte avant de prédire.")
    else:
        # Prétraitement du texte
        seq = tokenizer.texts_to_sequences([user_text])
        padded_seq = pad_sequences(seq, maxlen=50)

        # Prédiction
        prediction = model.predict(padded_seq)
        pred_value = float(prediction[0][0])

        # Déterminer le sentiment et la confiance
        sentiment = "Positif" if pred_value > 0.5 else "Négatif"
        confidence = round(pred_value * 100, 2)

        # Stocker les résultats dans st.session_state
        st.session_state["prediction_done"] = True
        st.session_state["sentiment_label"] = sentiment
        st.session_state["confidence_score"] = confidence
        st.session_state["log_sent"] = False  # Réinitialiser le statut du log

        # Afficher le résultat à l'utilisateur
        st.success(f"**Résultat : {sentiment} ({confidence}%)**")

# Afficher les boutons de feedback après la prédiction
if st.session_state.get("prediction_done"):
    st.write("La prédiction est-elle correcte ?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("OK"):
            st.success("Merci pour votre retour. La prédiction a été validée.")
            st.session_state["log_sent"] = True  # Empêche le bouton "Pas OK" d'envoyer un log après "OK"
    
    with col2:
        if st.button("Pas OK"):
            if not st.session_state["log_sent"]:
                # Logger le tweet mal prédit dans Azure
                logger.warning(
                    f"Prédiction incorrecte signalée : "
                    f"Texte = '{user_text}', "
                    f"Résultat = {st.session_state['sentiment_label']}, "
                    f"Confiance = {st.session_state['confidence_score']}%"
                )
                st.session_state["log_sent"] = True
                st.error("Merci pour votre retour. La prédiction incorrecte a été signalée.")
            else:
                st.info("La prédiction incorrecte a déjà été signalée.")
