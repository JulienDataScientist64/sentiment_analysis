# src/models/common.py
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
import time

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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import mlflow

def log_classification_report(y_true, y_pred, labels=None, target_names=None, report_title="Classification Report"):
    """
    Génère et journalise un rapport de classification et une matrice de confusion avec MLflow.
    
    Args:
        y_true (array-like): Vérités terrain.
        y_pred (array-like): Prédictions du modèle.
        labels (list, optional): Liste des indices des classes. Par défaut, None.
        target_names (list, optional): Noms des classes. Par défaut, None.
        report_title (str): Titre pour le rapport.
    """
    # Générer le rapport de classification
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=False)
    text_report = classification_report(y_true, y_pred, target_names=target_names)
    print(f"\n{report_title} :\n{text_report}")

    # Sauvegarder le rapport avec un horodatage pour éviter les conflits
    timestamp = int(time.time())
    report_path = f"classification_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(text_report)

    # Journaliser le rapport texte dans MLflow
    mlflow.log_artifact(report_path)
    print(f"Rapport de classification enregistré dans MLflow : {report_path}")

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    # Afficher et enregistrer la matrice de confusion
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"{report_title} - Matrice de Confusion")
    plt.show()

    cm_path = f"confusion_matrix_{timestamp}.png"
    fig.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path)
    print(f"Matrice de confusion enregistrée dans MLflow : {cm_path}")



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow

def plot_and_log_confusion_matrix(y_true, y_pred, labels=None, title="Matrice de Confusion"):
    """
    Trace et journalise une matrice de confusion avec MLflow.
    
    Args:
        y_true (array-like): Vérités terrain.
        y_pred (array-like): Prédictions du modèle.
        labels (list, optional): Liste des noms des classes. Par défaut, None.
        title (str): Titre de la matrice de confusion.
    """
    # Calcul de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Tracé de la matrice de confusion
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(title)
    plt.show()
    
    # Générer un nom de fichier unique avec horodatage
    timestamp = int(time.time())
    cm_path = f"confusion_matrix_{timestamp}.png"
    
    # Journaliser l'artefact
    fig.savefig(cm_path)
    plt.close(fig)  # Fermer la figure pour libérer de la mémoire
    mlflow.log_artifact(cm_path)
    print(f"Matrice de confusion enregistrée dans MLflow : {cm_path}")


