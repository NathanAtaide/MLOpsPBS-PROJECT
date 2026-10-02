"""Automatic retraining of the optimized model (run by the Airflow `fitcheck_model_retrain` DAG).

Steps:
1. Load data/fitcheck_dataset.csv (Kaggle base rows + every random addition so far).
2. Split off the notebook's shared hold-out (base rows only), so scores are comparable
   with the notebook's `holdout_eval_*` runs and with every previous retrain.
3. Refit the model currently served by the API with the same hyperparameters
   (`clone`), on all remaining rows including the new ones.
4. Score the new model and the current one ("champion") on the hold-out.
5. Log the run to MLflow (same metric names as the notebook). If the new model is not
   worse than the champion, register it as a new version of
   FitCheck_GBClassifier_optimized with the alias "champion" and save it to Models/.
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from fitcheck import dataset, pipeline_state
from fitcheck.evaluation import classification_metrics
from fitcheck.features import RAW_FEATURES, add_fit_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("MODEL_DIR", PROJECT_ROOT / "Models"))
MODEL_PATH = MODEL_DIR / "fitcheck_gb_model_optimized.joblib"   # the file the API serves
SRC_PACKAGE = Path(__file__).resolve().parent                    # shipped with the MLflow model

EXPERIMENT_NAME = os.getenv("FITCHECK_EXPERIMENT", "FitCheck_Sizing_Model_Docker")  # same as the notebook
REGISTERED_MODEL = "FitCheck_GBClassifier_optimized"
CHAMPION_ALIAS = "champion"
# Promote when the new macro-F1 is at least (champion macro-F1 - tolerance)
PROMOTION_TOLERANCE = float(os.getenv("FITCHECK_PROMOTION_TOLERANCE", "0.005"))

# Used only if no optimized model exists yet (the notebook's optimized cell was never run)
DEFAULT_MODEL_PARAMS = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.01,
    "class_weight": None,
}
LOGGED_PARAMS = ["learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization", "class_weight", "max_iter"]


def make_optimized_pipeline(**model_params):
    """Same pipeline as the notebook's optimized cell: features -> HistGradientBoosting."""
    return Pipeline([
        ("features", FunctionTransformer(add_fit_features)),
        ("model", HistGradientBoostingClassifier(
            categorical_features=["product_type_id"],
            max_iter=1000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            random_state=42, **model_params,
        )),
    ])


def _save_model_atomically(model):
    """Write to a temp file and rename, so the API never loads a half-written model."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(model, tmp)
    os.replace(tmp, MODEL_PATH)


def retrain(dag_run_id=None):
    """Retrain, evaluate, log to MLflow and (if good enough) promote. Returns a summary dict."""
    started = time.perf_counter()

    # ------------------------------------------------------------------
    # 1-2. Dataset snapshot + fixed hold-out
    # ------------------------------------------------------------------
    df = dataset.load_dataset(create_if_missing=False)
    additions_included = int(df["addition_id"].max())
    base = df[df["addition_id"] == 0]
    holdout_idx = dataset.shared_holdout_index(base[dataset.TARGET])  # positions == labels (base rows come first)
    train = df.drop(index=holdout_idx)
    X_train, y_train = train[RAW_FEATURES], train[dataset.TARGET]
    X_holdout, y_holdout = df.loc[holdout_idx, RAW_FEATURES], df.loc[holdout_idx, dataset.TARGET].to_numpy()

    # Hard cases: true clearance within 2 cm of a size boundary (same definition as the notebook)
    delta = dataset.chest_delta(*(df.loc[holdout_idx, c].to_numpy()
                                  for c in ["height_cm", "weight_kg", "garment_chest_cm", "fabric_stretch_pct"]))
    near_boundary = np.abs(np.abs(delta) - dataset.GOOD_FIT_TOLERANCE_CM) < 2.0

    # ------------------------------------------------------------------
    # 3. Refit the current champion's configuration on the updated data
    # ------------------------------------------------------------------
    champion = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    model = clone(champion) if champion is not None else make_optimized_pipeline(**DEFAULT_MODEL_PARAMS)
    fit_started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_started

    # ------------------------------------------------------------------
    # 4. Evaluate new model vs champion on the same hold-out
    # ------------------------------------------------------------------
    def evaluate(m):
        proba = m.predict_proba(X_holdout)
        return classification_metrics(y_holdout, m.classes_[proba.argmax(axis=1)], proba=proba,
                                      near_boundary=near_boundary)

    new_metrics = evaluate(model)
    champion_metrics = evaluate(champion) if champion is not None else None
    promoted = champion_metrics is None or new_metrics["f1_macro"] >= champion_metrics["f1_macro"] - PROMOTION_TOLERANCE

    # ------------------------------------------------------------------
    # 5. MLflow: run + (if promoted) new registered version with alias "champion"
    # ------------------------------------------------------------------
    mlflow.set_experiment(EXPERIMENT_NAME)
    hgb = model.named_steps["model"]
    added_rows = int((train["addition_id"] > 0).sum())
    user_rows = int((train["source"] == "user").sum())   # real feedback from users

    with mlflow.start_run(run_name=f"retrain_addition_{additions_included:04d}") as run:
        mlflow.set_tags({
            "model_variant": "optimized",   # same tags as the notebook runs -> easy filtering
            "eval_set": "shared_holdout",
            "run_type": "retrain",
            "trigger": "airflow" if dag_run_id else "manual",
            "airflow_dag_run_id": dag_run_id or "",
            "promoted": str(promoted).lower(),
        })
        params = {k: v for k, v in hgb.get_params().items() if k in LOGGED_PARAMS}
        mlflow.log_params({
            **params,
            "boosting_iterations": hgb.n_iter_,
            "train_rows": len(X_train),
            "eval_rows": len(X_holdout),
            "base_rows": len(base),
            "added_rows_in_training": added_rows,
            "user_rows_in_training": user_rows,
            "additions_included": additions_included,
            "promotion_tolerance": PROMOTION_TOLERANCE,
            "hyperparameters_from": "served champion" if champion is not None else "defaults",
        })
        # Same metric names as the notebook (accuracy, f1_macro, balanced_accuracy, recall_*, ...)
        mlflow.log_metrics(new_metrics)
        mlflow.log_metric("fit_seconds", fit_seconds)
        if champion_metrics is not None:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
            mlflow.log_metric("delta_f1_macro_vs_champion", new_metrics["f1_macro"] - champion_metrics["f1_macro"])

        # Dataset lineage: which data this model was trained on (shows up in the run's "Datasets")
        try:
            mlflow.log_input(mlflow.data.from_pandas(train, source=str(dataset.DATASET_PATH),
                                                     name="fitcheck_dataset", targets=dataset.TARGET),
                             context="training")
        except Exception as exc:  # lineage is nice-to-have; never fail a retrain because of it
            print(f"[fitcheck] could not log dataset lineage: {exc}")

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            signature=infer_signature(X_holdout.head(200), model.predict(X_holdout.head(200))),
            input_example=X_holdout.head(3),
            code_paths=[str(SRC_PACKAGE)],
            # Only a promoted model becomes a new registry version
            registered_model_name=REGISTERED_MODEL if promoted else None,
        )
        version = None
        if promoted:
            client = MlflowClient()
            version = getattr(model_info, "registered_model_version", None)
            if version is None:  # fallback: find the version created from this run
                found = client.search_model_versions(f"run_id='{run.info.run_id}'")
                version = found[0].version if found else None
        if version is not None:
            client.set_registered_model_alias(REGISTERED_MODEL, CHAMPION_ALIAS, version)
            client.set_model_version_tag(REGISTERED_MODEL, version, "trained_by", "airflow" if dag_run_id else "manual")
            client.set_model_version_tag(REGISTERED_MODEL, version, "additions_included", str(additions_included))

    # ------------------------------------------------------------------
    # 6. Serve it (if promoted) and record the retrain in the pipeline state
    # ------------------------------------------------------------------
    if promoted:
        _save_model_atomically(model)

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "airflow_dag_run_id": dag_run_id,
        "mlflow_run_id": run.info.run_id,
        "registered_version": str(version) if version is not None else None,
        "promoted": promoted,
        "additions_included": additions_included,
        "train_rows": len(X_train),
        "added_rows_in_training": added_rows,
        "user_rows_in_training": user_rows,
        "f1_macro": round(new_metrics["f1_macro"], 5),
        "accuracy": round(new_metrics["accuracy"], 5),
        "champion_f1_macro": round(champion_metrics["f1_macro"], 5) if champion_metrics else None,
        "seconds": round(time.perf_counter() - started, 1),
    }
    with pipeline_state.locked_state() as state:
        pipeline_state.record_retrain(state, summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(retrain(), indent=2))
