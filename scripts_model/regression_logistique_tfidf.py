import os
import pandas as pd
import joblib
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# --------------------------------------------------
# 1. Gestion des arguments en ligne de commande
# --------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Régression Logistique avec TF-IDF pour l'analyse de sentiments.")
    parser.add_argument("--data_path", type=str, required=True, help="Chemin vers le fichier CSV contenant les données.")
    parser.add_argument("--output_dir", type=str, required=True, help="Répertoire de sortie pour sauvegarder les modèles et rapports.")
    parser.add_argument("--max_features", type=int, default=5000, help="Nombre maximal de features pour le TF-IDF.")
    return parser.parse_args()

# --------------------------------------------------
# 2. Fonction principale
# --------------------------------------------------
def main():
    args = parse_arguments()

    # Configuration MLflow
    mlflow.set_tracking_uri(f"file:{args.output_dir}/mlruns")
    mlflow.set_experiment("Sentiment_Analysis_Models")

    # Chargement des données
    print(f"Chargement des données depuis {args.data_path}...")
    balanced_data = pd.read_csv(args.data_path)[['target', 'TweetText']]

    # Split des données
    train_data, temp_data = train_test_split(
        balanced_data, test_size=0.3, random_state=42, stratify=balanced_data['target']
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42, stratify=temp_data['target']
    )

    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(max_features=args.max_features)
    X_train_tfidf = vectorizer.fit_transform(train_data['TweetText'])
    y_train = train_data['target']
    X_val_tfidf = vectorizer.transform(val_data['TweetText'])
    y_val = val_data['target']
    X_test_tfidf = vectorizer.transform(test_data['TweetText'])
    y_test = test_data['target']

    # GridSearchCV
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear'],
        'max_iter': [100, 500, 1000]
    }
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42), param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1
    )

    # Entraînement + MLflow tracking
    with mlflow.start_run(run_name="TF-IDF_RegressionLogistique"):
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", args.max_features)
        mlflow.sklearn.autolog()

        grid_search.fit(X_train_tfidf, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        mlflow.log_params(best_params)

        # Validation
        y_val_pred = best_model.predict(X_val_tfidf)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("val_f1_score", val_f1)

        # Test
        y_test_pred = best_model.predict(X_test_tfidf)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_score", test_f1)

        # Rapport de classification
        report = classification_report(y_test, y_test_pred)
        report_path = os.path.join(args.output_dir, "classification_report_LogReg.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Sauvegarde du modèle et du vectoriseur
        model_path = os.path.join(args.output_dir, "tfidf_logistic_regression_best.pkl")
        vect_path = os.path.join(args.output_dir, "tfidf_vectorizer.pkl")
        joblib.dump(best_model, model_path)
        joblib.dump(vectorizer, vect_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(vect_path)

        print("\nValidation Accuracy :", val_accuracy)
        print("Validation F1 Score :", val_f1)
        print("\nTest Accuracy :", test_accuracy)
        print("Test F1 Score :", test_f1)
        print("\nClassification Report (Test):\n", report)

    print(f"[INFO] Entraînement et évaluation terminés. Les logs sont disponibles dans {args.output_dir}/mlruns.")

# --------------------------------------------------
# 3. Exécution du script
# --------------------------------------------------
if __name__ == "__main__":
    main()
