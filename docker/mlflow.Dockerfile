# MLflow tracking server + model registry (used by docker-compose.yml)
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install "mlflow==3.18.0"
WORKDIR /mlflow
EXPOSE 5000
