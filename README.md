Sentiment Analysis Project
==========================

Objectif du projet
------------------
L'objectif principal de ce projet est de détecter et analyser le sentiment exprimé dans des tweets ou textes, tout en permettant une évaluation continue des prédictions incorrectes fournies par le modèle. Ce projet inclut :

- Le développement d'un modèle de Machine Learning (LSTM) pour l'analyse des sentiments.
- Le déploiement du modèle sous forme d'une API accessible.
- L'intégration de mécanismes de suivi des erreurs via Azure Application Insights.
- La collecte des tweets mal prédits pour améliorer le modèle au fil du temps.

Structure du projet
-------------------
    sentiment_analysis/
        README.md               : Ce fichier expliquant le projet
        pyproject.toml          : Liste des packages nécessaires pour POETRY
        poetry.lock             : Verrouille les versions des packages et leurs dépendances
        Dockerfile              : Fichier pour le déploiement conteneurisé
        heroku.yml              : Fichier pour le déploiement sur Heroku
        notebooks/              : Notebooks pour l'exploration et la modélisation

    app/                    : Code source de l'API
        main.py             : Point d'entrée principal de l'API
        app_local.py        : Prédiction STREAMLIT en local
        app_cloud.py        : Prédiction STREAMLIT cloud de l'API
        log_insights.py     : Suivi des performances sur AZURE

    Mlruns/                 : Suivi des modélisations sur MLFLOW

    models/                 : Modèles préentraînés et tokenizer
        lstm.h5             : Modèle au format H5
        lstm.pkl            : Modèle au format PKL
        tokenizer.pkl       : Tokenizer associé

    tests/                  : Tests unitaires et d'intégration (PYTEST)
        test_api.py         : Tests de l'API
        test_pred.py        : Tests des prédictions

    .github/workflows       : Workflows CI/CD
        deploy.yaml         : Déploiement continu avec GitHub Actions
