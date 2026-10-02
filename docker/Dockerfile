# MLOPS Project - DOCKER
#
# One image, four roles (selected by the `command` in docker-compose.yml):
#   - api     -> FastAPI inference service + web UI      (port 3000)
#   - mlflow  -> MLflow tracking server + registry       (port 5001)
#   - jupyter -> JupyterLab for training/experiments     (port 8888)
#   - airflow -> Airflow: data ingestion + retrain DAGs  (port 8080)
# Using a single image keeps library versions identical between the place a
# model is trained (Jupyter/Airflow), where it is tracked (MLflow) and where it is served (API).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/Models \
    PYTHONPATH=/app/src \
    AIRFLOW_HOME=/opt/airflow

# Define a pasta de trabalho dentro do container
WORKDIR /app

# --- Airflow, in its OWN virtualenv -------------------------------------------------
# Airflow's official constraints pin scikit-learn/pandas versions that conflict with the
# scikit-learn 1.6.1 our saved models need, so it cannot share the main environment.
# The DAGs run the ML code with the main Python (/usr/local/bin/python) instead.
# This layer comes first because it is the heaviest and changes least often.
ARG AIRFLOW_VERSION=3.3.2
ARG AIRFLOW_PYTHON=3.13
RUN python -m venv /opt/airflow-venv \
    && /opt/airflow-venv/bin/pip install --no-cache-dir \
        "apache-airflow==${AIRFLOW_VERSION}" \
        apache-airflow-providers-standard \
        --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${AIRFLOW_PYTHON}.txt"

# Copia apenas o arquivo de requisitos primeiro (isso acelera futuros builds)
COPY docker/requirements.txt .

# Instala as dependências listadas
RUN pip install --no-cache-dir -r requirements.txt

# Copia as pastas essenciais do projeto para dentro da imagem
# (Models is baked in so the API image works standalone; compose mounts the
#  live folder on top so models retrained in Jupyter/Airflow are picked up.)
COPY api /app/api
COPY src /app/src
COPY Models /app/Models
COPY airflow/dags /app/airflow/dags

# Portas: API, MLflow, Jupyter, Airflow
EXPOSE 3000 5001 8888 8080

# Default role = API. python is used for the probe because curl is not in the slim image.
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "3000"]
