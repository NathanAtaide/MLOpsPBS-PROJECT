"""Command-line entry point for the data/retraining pipeline.

The Airflow DAGs call these commands with the project's main Python (the one with
scikit-learn 1.6.1 and MLflow), because Airflow runs in its own virtualenv. The same
commands can be run by hand to test a step without Airflow:

    python -m fitcheck.pipeline ingest --rows 100   # add one batch of random rows
    python -m fitcheck.pipeline retrain             # retrain + evaluate + log to MLflow
    python -m fitcheck.pipeline status              # additions, pending count, last retrains
"""

import argparse
import json
from datetime import datetime, timezone

from fitcheck import pipeline_state


def ingest(rows, run_id=None):
    """Append one addition (a batch of random labelled rows) to the dataset."""
    import pandas as pd  # heavy imports only needed here

    from fitcheck import dataset

    with pipeline_state.locked_state() as state:
        df = dataset.load_dataset()
        # The dataset file is the source of truth for ids (safe even if a previous run died
        # after writing the CSV but before updating the state)
        addition_id = max(int(df["addition_id"].max()), state["last_addition_id"]) + 1
        base = df[df["addition_id"] == 0]
        new_rows = dataset.generate_random_records(rows, base, addition_id)
        dataset.save_dataset(pd.concat([df, new_rows], ignore_index=True))

        state["last_addition_id"] = addition_id
        state["last_addition_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        pending = pipeline_state.pending_additions(state)

    return {
        "addition_id": addition_id,
        "rows_added": len(new_rows),
        "dataset_rows": len(df) + len(new_rows),
        "class_counts_added": {int(k): int(v) for k, v in new_rows[dataset.TARGET].value_counts().sort_index().items()},
        "pending_additions": pending,
        "additions_per_retrain": pipeline_state.additions_per_retrain(),
        "airflow_run_id": run_id,
    }


def status():
    state = pipeline_state.read_state()
    return {
        **{k: v for k, v in state.items() if k != "retrains"},
        "pending_additions": pipeline_state.pending_additions(state),
        "additions_per_retrain": pipeline_state.additions_per_retrain(),
        "last_retrains": state["retrains"][-5:],
    }


def main():
    parser = argparse.ArgumentParser(prog="python -m fitcheck.pipeline", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="append a batch of random rows to the dataset")
    p_ingest.add_argument("--rows", type=int, default=100, help="rows per addition (default 100)")
    p_ingest.add_argument("--run-id", default=None, help="Airflow run id, for traceability")

    p_retrain = sub.add_parser("retrain", help="retrain, evaluate, log to MLflow, promote if good")
    p_retrain.add_argument("--run-id", default=None, help="Airflow run id, for traceability")

    sub.add_parser("status", help="show pipeline counters and recent retrains")

    args = parser.parse_args()
    if args.command == "ingest":
        result = ingest(args.rows, args.run_id)
    elif args.command == "retrain":
        from fitcheck.training import retrain  # heavy imports only when retraining
        result = retrain(dag_run_id=args.run_id)
    else:
        result = status()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
