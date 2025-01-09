# Utiliser une image officielle Python
FROM python:3.10-slim

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Ajouter Poetry au PATH
ENV PATH="/root/.local/bin:$PATH"

# Copier les fichiers de configuration pour installer les dépendances
COPY pyproject.toml poetry.lock /app/

# Installer les dépendances
RUN poetry install --no-interaction --no-ansi

# Copier le reste du code
COPY . /app/

# Exposer le port 8000
EXPOSE 8000

# Commande pour lancer l'application FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
