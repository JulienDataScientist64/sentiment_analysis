
import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Charger le modèle
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/final_trained_light_LSTM.h5")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

# Interface utilisateur
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
        st.write(f"**Résultat : {sentiment}**")
    else:
        st.write("⚠️ Veuillez entrer un texte.")
