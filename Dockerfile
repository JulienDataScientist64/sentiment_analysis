# Utiliser l'image Python officielle comme image de base
FROM python:3.10-slim

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app  

# Définir le répertoire de travail
WORKDIR /app

# Mettre à jour pip et installer Poetry
RUN pip install --upgrade pip \
    && pip install poetry==1.6.1

# Configurer Poetry pour ne pas créer de virtualenv
RUN poetry config virtualenvs.create false

# Copier uniquement les fichiers de configuration en premier pour tirer parti du cache Docker
COPY pyproject.toml poetry.lock /app/

# Installer les dépendances
RUN poetry install --no-interaction --no-ansi

# Copier le reste du code de l'application
COPY . /app/

# Exposer le port 8000
EXPOSE 8000

# Lancer l'application FastAPI avec Uvicorn
ENTRYPOINT ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
