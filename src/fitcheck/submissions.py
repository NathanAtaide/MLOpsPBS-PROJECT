"""Real data from users: the API queues it, the Airflow ingestion DAG consumes it.

Why a queue file instead of writing straight into the dataset?
The API and Airflow run in different containers. The dataset and the pipeline state are
protected by a file lock that only works *inside* the Airflow container, so only Airflow is
allowed to change them. The API only ever APPENDS one line to data/user_submissions.jsonl
(both containers see the same ./data folder); the next ingestion run reads the new lines.

Standard library only on purpose, like pipeline_state.py: it is imported by the API, by the
pipeline CLI and (indirectly) by the Airflow DAGs.

Each line is one JSON object:
    {"height_cm": 175, "weight_kg": 72, "garment_chest_cm": 96, "fabric_stretch_pct": 3,
     "product_type_id": 0, "actual_fit": 1, "submitted_at": "2026-10-02T20:00:00+00:00"}
`actual_fit` is what the user really experienced: 0 = Too Small, 1 = Good Fit, 2 = Too Large.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FITCHECK_DATA_DIR", PROJECT_ROOT / "data"))
SUBMISSIONS_PATH = DATA_DIR / "user_submissions.jsonl"

REQUIRED_KEYS = ("height_cm", "weight_kg", "garment_chest_cm", "fabric_stretch_pct",
                 "product_type_id", "actual_fit")


def max_rows_per_addition():
    """Upper bound of user rows folded into one addition (default 5000)."""
    return int(os.getenv("FITCHECK_MAX_USER_ROWS_PER_ADDITION", "5000"))


def append_submission(record):
    """Queue one submission. Returns its (1-based) line number in the queue."""
    line = json.dumps({**record, "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                      separators=(",", ":")) + "\n"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Append mode: concurrent writers add lines instead of overwriting each other
    with open(SUBMISSIONS_PATH, "ab") as f:
        f.write(line.encode("utf-8"))
    return count_lines()


def _complete_lines():
    """Every complete line of the queue (a half-written last line is ignored until finished)."""
    if not SUBMISSIONS_PATH.exists():
        return []
    raw = SUBMISSIONS_PATH.read_bytes()
    lines = raw.split(b"\n")
    return lines[:-1]  # whatever follows the last "\n" is incomplete (or empty)


def count_lines():
    """How many submissions have ever been queued."""
    return len(_complete_lines())


def read_submissions(skip=0, limit=None):
    """Valid submissions after the first `skip` lines.

    Returns (records, lines_taken). `lines_taken` counts malformed lines too, so the caller
    can advance its "consumed lines" counter by exactly that amount and never re-read them.
    """
    lines = _complete_lines()[skip:]
    if limit is not None:
        lines = lines[:limit]
    records = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and all(k in item for k in REQUIRED_KEYS):
            records.append(item)
    return records, len(lines)
