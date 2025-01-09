import pytest
from fastapi.testclient import TestClient
from app.main import app

# Crée un client de test FastAPI
client = TestClient(app)

# Test de la route racine
def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert "API is running" in res.json()["message"]

# Test de la route /predict
def test_predict():
    payload = [[5.1, 3.5, 1.4, 0.2]]
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "predictions" in res.json()
