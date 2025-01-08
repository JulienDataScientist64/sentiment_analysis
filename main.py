from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur mon API de sentiment analysis 🚀"}

@app.get("/predict/{text}")
def predict_sentiment(text: str):
    if "bien" in text.lower():
        return {"text": text, "sentiment": "positif"}
    else:
        return {"text": text, "sentiment": "négatif"}
