import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test de la route principale
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "API is running" in response.json()["message"]

# Test de la route /predict avec un exemple de tweet
def test_predict():
    # Envoie une liste de textes directement, comme attendu par la route
    payload = ["I love this product! It's amazing!"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert isinstance(response.json()["predictions"], list)
