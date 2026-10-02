"""Pipeline bookkeeping: how many additions happened and when the model was last retrained.

Standard library only on purpose: this module is imported both by the training code
(main Python environment) and by the Airflow DAGs (Airflow's own virtualenv, which has no
pandas/scikit-learn).

State lives in data/pipeline_state.json:
    last_addition_id                  id of the newest addition in the dataset
    last_retrained_addition_id        newest addition included in the last finished retrain
    consumed_submission_lines         how many lines of data/user_submissions.jsonl were already
                                      turned into dataset rows
    retrain_requested_at_addition_id  set when the ingestion DAG triggers a retrain, so the
                                      next ingestions don't trigger it again while it runs
    retrains                          short history of retrain results (newest last)
"""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FITCHECK_DATA_DIR", PROJECT_ROOT / "data"))
STATE_PATH = DATA_DIR / "pipeline_state.json"
# Lock file on the container's own filesystem (file locks are unreliable on Windows bind mounts).
# Every process that changes the state runs inside the Airflow container, so this is enough.
LOCK_PATH = Path(os.getenv("FITCHECK_LOCK_FILE", "/tmp/fitcheck_pipeline.lock"))

HISTORY_LENGTH = 20
DEFAULT_STATE = {
    "last_addition_id": 0,
    "last_addition_at": None,
    "last_retrained_addition_id": 0,
    "consumed_submission_lines": 0,
    "retrain_requested_at_addition_id": None,
    "retrains": [],
}


def additions_per_retrain():
    """How many additions trigger a retrain (default 50)."""
    return int(os.getenv("FITCHECK_ADDITIONS_PER_RETRAIN", "50"))


def read_state():
    """Current state (defaults if the pipeline never ran)."""
    if not STATE_PATH.exists():
        return json.loads(json.dumps(DEFAULT_STATE))
    return {**DEFAULT_STATE, **json.loads(STATE_PATH.read_text(encoding="utf-8"))}


def _write_state(state):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, STATE_PATH)  # atomic: readers see the old or the new file, never half of one


@contextmanager
def locked_state():
    """Read-modify-write the state under an exclusive lock; changes are saved on exit."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = read_state()
            yield state
            _write_state(state)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def pending_additions(state):
    """Additions not yet covered by a finished or already-requested retrain."""
    covered = max(state["last_retrained_addition_id"], state["retrain_requested_at_addition_id"] or 0)
    return state["last_addition_id"] - covered


def record_retrain(state, summary):
    """Mark additions up to summary['additions_included'] as consumed and keep the history short."""
    consumed = summary["additions_included"]
    state["last_retrained_addition_id"] = max(state["last_retrained_addition_id"], consumed)
    requested = state["retrain_requested_at_addition_id"]
    if requested is not None and requested <= consumed:
        state["retrain_requested_at_addition_id"] = None
    state["retrains"] = (state["retrains"] + [summary])[-HISTORY_LENGTH:]
