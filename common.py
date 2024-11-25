# src/models/common.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np

def log_metrics(model, X_test, y_test):
    """
    Journalise les métriques d'évaluation pour l'ensemble de test.
    Args:
        model: Modèle entraîné (Keras ou scikit-learn).
        X_test: Features de test.
        y_test: Labels de test.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):  # Modèles scikit-learn
        y_proba = model.predict_proba(X_test)[:, 1]
    else:  # Modèles Keras
        y_proba = model.predict(X_test).flatten()

    metrics = {
        "accuracy_test": accuracy_score(y_test, (y_pred > 0.5).astype(int) if isinstance(y_pred, np.ndarray) else y_pred),
        "f1_test": f1_score(y_test, (y_pred > 0.5).astype(int) if isinstance(y_pred, np.ndarray) else y_pred),
        "auc_roc_test": roc_auc_score(y_test, y_proba),
    }

    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def plot_and_log_accuracy_loss(history, model_name):
    """
    Trace et enregistre les courbes d'accuracy et de perte.
    Args:
        history: Historique d'entraînement Keras.
        model_name: Nom du modèle (pour nommer les fichiers).
    """
    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Train vs Validation Accuracy')
    accuracy_path = f"accuracy_plot_{model_name}.png"
    plt.savefig(accuracy_path)
    mlflow.log_artifact(accuracy_path)
    plt.close()

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Train vs Validation Loss')
    loss_path = f"loss_plot_{model_name}.png"
    plt.savefig(loss_path)
    mlflow.log_artifact(loss_path)
    plt.close()
