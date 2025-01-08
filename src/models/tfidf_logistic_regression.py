import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from azureml.core import Workspace
from src.preprocessing import load_and_preprocess_data  # Importation du fichier preprocessing.py


def train_tfidf_logistic_regression(file_path, experiment_name="Sentiment Analysis TF-IDF", max_features=5000):
    """
    Entraîne un modèle de régression logistique avec TF-IDF et journalise les résultats dans MLflow.
    Args:
        file_path (str): Chemin vers le fichier contenant les données.
        experiment_name (str): Nom de l'expérience dans MLflow.
        max_features (int): Nombre maximum de features pour TF-IDF.
    """
    # Chargement et configuration de l'AzureML Workspace
    try:
        ws = Workspace.from_config(path='/content/drive/MyDrive/AzureML/config.json')
        mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
        mlflow.set_experiment(experiment_name)
        print(f"Workspace AzureML chargé : {ws.name}")
    except Exception as e:
        print("Erreur de connexion au Workspace AzureML :", e)
        return

    # Chargement et prétraitement des données depuis preprocessing.py
    print("Chargement et prétraitement des données...")
    df = load_and_preprocess_data(file_path, num_positive=100000, num_negative=100000)

    # Vectorisation avec TF-IDF
    print("Vectorisation avec TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X = tfidf_vectorizer.fit_transform(df['TweetText'])
    y = df['sentiment']

    # Division des données
    print("Division des données en ensembles d'entraînement, de validation et de test...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Définition des hyperparamètres
    param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Entraîner et journaliser avec MLflow
    with mlflow.start_run(run_name="TF-IDF_Logistic_Regression"):
        # Initialisation et recherche d'hyperparamètres
        print("Recherche des meilleurs hyperparamètres avec GridSearchCV...")
        model = LogisticRegression(max_iter=300)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=cv, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Meilleurs paramètres
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        # Évaluation sur validation
        print("Évaluation sur ensemble de validation...")
        y_val_pred = best_model.predict(X_val)
        mlflow.log_metrics({
            "accuracy_val": accuracy_score(y_val, y_val_pred),
            "f1_val": f1_score(y_val, y_val_pred, pos_label=1),
            "auc_roc_val": roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1]),
        })

        # Évaluation sur test
        print("Évaluation sur ensemble de test...")
        y_test_pred = best_model.predict(X_test)
        mlflow.log_metrics({
            "accuracy_test": accuracy_score(y_test, y_test_pred),
            "f1_test": f1_score(y_test, y_test_pred, pos_label=1),
            "auc_roc_test": roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]),
        })

        # Enregistrement du modèle et des artefacts
        print("Enregistrement du modèle et des artefacts...")
        input_example = X_test[:1]
        signature = infer_signature(input_example, best_model.predict(input_example))
        mlflow.sklearn.log_model(best_model, "model", signature=signature, input_example=input_example)

        print(f"Run enregistré avec MLflow. ID du Run: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    # Spécifier le chemin du fichier CSV et lancer l'entraînement
    train_tfidf_logistic_regression(
        file_path="/content/drive/MyDrive/Réalisez une analyse de sentiments grâce au Deep Learning/training.csv",
        experiment_name="Sentiment Analysis TF-IDF"
    )
