"""FitCheck data + retraining pipeline.

Two DAGs:

fitcheck_data_ingestion  (every minute, starts PAUSED: switch it on in the Airflow UI)
    add_random_rows  ->  retrain_threshold_reached  ->  trigger_retrain
    Each run is one "addition": a batch of random labelled rows appended to
    data/fitcheck_dataset.csv. After 50 additions (FITCHECK_ADDITIONS_PER_RETRAIN) the
    short-circuit lets the trigger through and the retrain DAG starts.

fitcheck_model_retrain  (no schedule: started by the trigger, or by hand in the UI)
    retrain_and_evaluate  ->  reload_api
    Refits the served model on the updated dataset, scores it on the notebook's hold-out,
    logs everything to MLflow (http://localhost:5001), registers + promotes it when it is
    at least as good as the current model, then tells the API to load the new file.

Airflow runs in its own virtualenv, so the heavy work (pandas / scikit-learn / MLflow) is
done by `python -m fitcheck.pipeline ...` using the project's main Python. This file only
uses the standard library and fitcheck.pipeline_state (also standard library only).
"""

import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone

from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Param, dag, task

from fitcheck import pipeline_state

# Main Python of the image (scikit-learn 1.6.1 + MLflow), not Airflow's virtualenv
FITCHECK_PYTHON = os.getenv("FITCHECK_PYTHON", "/usr/local/bin/python")
ROWS_PER_ADDITION = int(os.getenv("FITCHECK_ROWS_PER_ADDITION", "100"))
INGEST_EVERY_MINUTES = int(os.getenv("FITCHECK_INGEST_EVERY_MINUTES", "1"))
API_URL = os.getenv("FITCHECK_API_URL", "http://api:3000")

INGESTION_DAG_ID = "fitcheck_data_ingestion"
RETRAIN_DAG_ID = "fitcheck_model_retrain"
START_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)
TAGS = ["fitcheck", "mlops"]


# ---------------------------------------------------------------------------
# DAG 1: add random data, trigger retraining every N additions
# ---------------------------------------------------------------------------
@dag(
    dag_id=INGESTION_DAG_ID,
    schedule=timedelta(minutes=INGEST_EVERY_MINUTES),
    start_date=START_DATE,
    catchup=False,          # don't backfill every minute since START_DATE
    max_active_runs=1,      # additions happen one at a time
    is_paused_upon_creation=True,  # nothing is added until you switch it on in the UI
    params={"rows_per_addition": Param(ROWS_PER_ADDITION, type="integer", minimum=1, maximum=10_000,
                                       description="Random rows appended per addition")},
    default_args={"retries": 1, "retry_delay": timedelta(seconds=20)},
    tags=TAGS,
    doc_md=__doc__,
)
def fitcheck_data_ingestion():
    # 1. One addition = one batch of random rows (labelled with the same rule as the notebook)
    add_random_rows = BashOperator(
        task_id="add_random_rows",
        bash_command=(FITCHECK_PYTHON + " -m fitcheck.pipeline ingest"
                      " --rows {{ params.rows_per_addition }} --run-id '{{ run_id }}'"),
    )

    # 2. Continue only when enough additions have accumulated since the last retrain
    @task.short_circuit()
    def retrain_threshold_reached():
        threshold = pipeline_state.additions_per_retrain()
        with pipeline_state.locked_state() as state:
            pending = pipeline_state.pending_additions(state)
            print(f"{pending}/{threshold} additions since the last retrain "
                  f"(last addition id: {state['last_addition_id']})")
            if pending < threshold:
                return False  # downstream trigger is skipped
            # Remember that a retrain was requested, so the next runs don't trigger it again
            state["retrain_requested_at_addition_id"] = state["last_addition_id"]
            return True

    # 3. Fire the retrain DAG (does not wait: ingestion keeps its 1-minute rhythm)
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id=RETRAIN_DAG_ID,
        conf={"requested_by": "{{ run_id }}"},
        wait_for_completion=False,
    )

    add_random_rows >> retrain_threshold_reached() >> trigger_retrain


# ---------------------------------------------------------------------------
# DAG 2: retrain, evaluate, register in MLflow, refresh the API
# ---------------------------------------------------------------------------
@dag(
    dag_id=RETRAIN_DAG_ID,
    schedule=None,          # triggered by the ingestion DAG (or "Trigger" in the UI)
    start_date=START_DATE,
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,  # must be active, otherwise triggered runs just wait
    tags=TAGS,
    doc_md=__doc__,
)
def fitcheck_model_retrain():
    # 1. Retrain + evaluate + MLflow logging/registration + save to Models/ if promoted
    retrain_and_evaluate = BashOperator(
        task_id="retrain_and_evaluate",
        bash_command=FITCHECK_PYTHON + " -m fitcheck.pipeline retrain --run-id '{{ run_id }}'",
        execution_timeout=timedelta(minutes=30),
    )

    # 2. Ask the running API to load the new model file (only needed when it was promoted)
    @task(retries=3, retry_delay=timedelta(seconds=15))
    def reload_api():
        retrains = pipeline_state.read_state()["retrains"]
        last = retrains[-1] if retrains else {}
        print("Retrain result:", json.dumps(last, indent=2))
        if not last.get("promoted"):
            print("New model was not promoted; the API keeps serving the current model.")
            return last

        request = urllib.request.Request(f"{API_URL}/reload", method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            served = json.load(response)
        print("API now serves:", json.dumps(served.get("models", {}), indent=2))
        return {**last, "api_optimized_model": served.get("models", {}).get("optimized")}

    retrain_and_evaluate >> reload_api()


fitcheck_data_ingestion()
fitcheck_model_retrain()
