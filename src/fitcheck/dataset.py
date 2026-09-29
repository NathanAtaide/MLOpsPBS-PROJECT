"""The FitCheck training dataset as a file that grows over time.

The notebook rebuilds the hybrid dataset in memory on every run. The Airflow pipeline
needs a persistent dataset it can append to, so this module:

* builds the *same* hybrid dataset as the notebook (same seed, same labelling rule), and
  stores it in data/fitcheck_dataset.csv with `addition_id = 0` for those base rows;
* generates new random records (addition 1, 2, 3, ...) labelled with the same rule;
* reproduces the notebook's shared hold-out split, so every retrained model is scored on
  exactly the rows the notebook's comparison used.
"""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fitcheck.features import BOUNDS, RAW_FEATURES, bmi

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("FITCHECK_DATA_DIR", PROJECT_ROOT / "data"))
KAGGLE_CSV = DATA_DIR / "final_test.csv"             # raw Kaggle file (also used by the notebook)
DATASET_PATH = DATA_DIR / "fitcheck_dataset.csv"     # growing training dataset

KAGGLE_DATASET = "tourist55/clothessizeprediction"
GOOD_FIT_TOLERANCE_CM = 4.0
TARGET = "fit_class"
COLUMNS = RAW_FEATURES + ["imc_index", TARGET, "addition_id", "added_at"]


# ---------------------------------------------------------------------------
# Labelling rule (identical to the notebook)
# ---------------------------------------------------------------------------
def chest_delta(height_cm, weight_kg, garment_chest_cm, fabric_stretch_pct):
    """Garment clearance in cm: stretch-adjusted garment chest minus estimated body chest."""
    estimated_body_chest = 0.45 * weight_kg + 0.35 * height_cm
    return (garment_chest_cm + fabric_stretch_pct * 0.5) - estimated_body_chest


def label_fit(delta):
    """0 = Too Small, 1 = Good Fit (within ±4 cm), 2 = Too Large."""
    return np.where(delta < -GOOD_FIT_TOLERANCE_CM, 0, np.where(delta > GOOD_FIT_TOLERANCE_CM, 2, 1))


def _synthetic_garments(rng, n):
    """Garment features with the same distributions as the notebook."""
    garment_chest_cm = np.clip(rng.normal(95.0, 15.0, size=n), *BOUNDS["garment_chest_cm"])
    fabric_stretch_pct = np.clip(rng.normal(5.0, 5.0, size=n), *BOUNDS["fabric_stretch_pct"])
    product_type_id = rng.choice([0, 1, 2], size=n)
    return garment_chest_cm, fabric_stretch_pct, product_type_id


def _frame(height_cm, weight_kg, garment_chest_cm, fabric_stretch_pct, product_type_id, addition_id, added_at):
    delta = chest_delta(height_cm, weight_kg, garment_chest_cm, fabric_stretch_pct)
    return pd.DataFrame({
        "height_cm": height_cm,
        "weight_kg": weight_kg,
        "garment_chest_cm": garment_chest_cm,
        "fabric_stretch_pct": fabric_stretch_pct,
        "product_type_id": product_type_id,
        "imc_index": bmi(height_cm, weight_kg),
        TARGET: label_fit(delta),
        "addition_id": addition_id,
        "added_at": added_at,
    })[COLUMNS]


# ---------------------------------------------------------------------------
# Base dataset (addition 0) = the notebook's hybrid dataset
# ---------------------------------------------------------------------------
def download_kaggle_csv():
    """Download the Kaggle CSV into data/ if it is not there yet (same as the notebook)."""
    if KAGGLE_CSV.exists():
        return KAGGLE_CSV
    import kagglehub

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    for file in os.listdir(path):
        if file.endswith(".csv"):
            shutil.copy(os.path.join(path, file), KAGGLE_CSV)
    return KAGGLE_CSV


def build_base_dataset():
    """Rebuild the notebook's hybrid dataset row for row.

    np.random.RandomState(42) produces the same numbers as the notebook's
    np.random.seed(42) + np.random.* calls, as long as the calls happen in the same order.
    """
    kaggle_raw = pd.read_csv(download_kaggle_csv()).dropna(subset=["height", "weight"])
    rng = np.random.RandomState(42)
    n = len(kaggle_raw)

    height_cm = np.clip(kaggle_raw["height"].values, *BOUNDS["height_cm"])
    weight_kg = np.clip(kaggle_raw["weight"].values, *BOUNDS["weight_kg"])
    garment_chest_cm, fabric_stretch_pct, product_type_id = _synthetic_garments(rng, n)
    return _frame(height_cm, weight_kg, garment_chest_cm, fabric_stretch_pct, product_type_id,
                  addition_id=0, added_at="")


# ---------------------------------------------------------------------------
# Random additions (addition 1, 2, 3, ...)
# ---------------------------------------------------------------------------
def generate_random_records(n, base, addition_id):
    """Create `n` new labelled records.

    Body measurements are drawn from real Kaggle people in the base data (+ a little noise,
    so they are new people rather than copies); garments come from the notebook's
    distributions. The RNG is seeded with the addition id, so every addition is reproducible.
    """
    rng = np.random.default_rng(1_000 + addition_id)
    people = base[["height_cm", "weight_kg"]].to_numpy()[rng.integers(0, len(base), size=n)]
    height_cm = np.clip(people[:, 0] + rng.normal(0.0, 2.0, size=n), *BOUNDS["height_cm"])
    weight_kg = np.clip(people[:, 1] + rng.normal(0.0, 2.0, size=n), *BOUNDS["weight_kg"])
    garment_chest_cm, fabric_stretch_pct, product_type_id = _synthetic_garments(rng, n)
    added_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return _frame(height_cm, weight_kg, garment_chest_cm, fabric_stretch_pct, product_type_id,
                  addition_id=addition_id, added_at=added_at)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def load_dataset(create_if_missing=True):
    """Read data/fitcheck_dataset.csv, creating it from the Kaggle data on first use."""
    if not DATASET_PATH.exists():
        if not create_if_missing:
            raise FileNotFoundError(f"{DATASET_PATH} does not exist yet; run an ingestion first")
        save_dataset(build_base_dataset())
    return pd.read_csv(DATASET_PATH, keep_default_na=False, dtype={"added_at": str})


def save_dataset(df):
    """Write atomically (temp file + rename), so a reader never sees a half-written CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DATASET_PATH.with_suffix(".csv.tmp")
    df[COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, DATASET_PATH)


# ---------------------------------------------------------------------------
# Shared hold-out (identical to the notebook's optimized/comparison cells)
# ---------------------------------------------------------------------------
def shared_holdout_index(base_labels):
    """Index of the hold-out rows within the BASE dataset (addition 0).

    Same calls and seeds as the notebook: exclude the rows the baseline trained on, then
    draw 20% of all base rows as hold-out. Added rows are never part of the hold-out, so
    every retrained model is scored on the same fixed rows and results stay comparable.
    """
    y = base_labels.reset_index(drop=True)
    sample_idx, _ = train_test_split(y.index, train_size=15000, random_state=42, stratify=y)
    baseline_train_idx, _ = train_test_split(sample_idx, test_size=0.2, random_state=42)
    candidates = y.index.difference(baseline_train_idx)
    _, holdout_idx = train_test_split(
        candidates, test_size=int(0.2 * len(y)), random_state=42, stratify=y.loc[candidates]
    )
    return holdout_idx
