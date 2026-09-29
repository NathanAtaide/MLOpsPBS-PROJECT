import hashlib
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from fitcheck.features import BOUNDS, CLASS_LABELS, RAW_FEATURES, bmi, build_baseline_features

app = FastAPI(title="FitCheck Microservice", version="1.1.0")

# Models live in MODEL_DIR (/app/Models in the image). Previously the service looked in
# the current directory, found nothing, and silently started with model=None.
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "Models"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

# uvicorn's logger, so every prediction shows up in `docker compose logs -f api`
log = logging.getLogger("uvicorn.error")

MODEL_FILES = {
    "baseline": "fitcheck_gb_model.joblib",
    "optimized": "fitcheck_gb_model_optimized.joblib",
}


def _load(filename):
    """Load a joblib artifact, returning None (and logging why) if it is missing or broken."""
    path = MODEL_DIR / filename
    try:
        return joblib.load(path)
    except Exception as exc:
        log.warning("[fitcheck] could not load %s: %s", path, exc)
        return None


def _fingerprint(filename):
    """Identify exactly which model file is being served: SHA-256 of its bytes + modified time.

    The notebook can compute the same hash for a file in Models/, so anyone can check that
    the predictions shown in the UI come from that specific trained model.
    """
    path = MODEL_DIR / filename
    return {
        "file": filename,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def load_models():
    """(Re)load every model from MODEL_DIR. Runs at startup and on POST /reload."""
    global baseline_model, baseline_features, optimized_model, MODELS, MODEL_INFO

    # Baseline: GradientBoostingClassifier trained on one-hot features (needs model_features.joblib).
    baseline_model = _load(MODEL_FILES["baseline"])
    baseline_features = _load("model_features.joblib") or []
    # Optimized: full sklearn Pipeline that takes the raw inputs directly (notebook or Airflow retrain).
    optimized_model = _load(MODEL_FILES["optimized"])

    models = {
        name: model
        for name, model in {"baseline": baseline_model, "optimized": optimized_model}.items()
        if model is not None
    }
    # Fingerprint + estimator type of every loaded model
    info = {
        name: {**_fingerprint(MODEL_FILES[name]), "estimator": type(model).__name__}
        for name, model in models.items()
    }
    # Swap both dicts only after everything loaded, so requests never see a half-reloaded state
    MODELS, MODEL_INFO = models, info
    for name, details in MODEL_INFO.items():
        log.info("[fitcheck] loaded %s model: %s", name, details)


load_models()


class PredictionInput(BaseModel):
    # Bounds match the ranges the training data was clipped to.
    height_cm: float = Field(ge=BOUNDS["height_cm"][0], le=BOUNDS["height_cm"][1])
    weight_kg: float = Field(ge=BOUNDS["weight_kg"][0], le=BOUNDS["weight_kg"][1])
    garment_chest_cm: float = Field(ge=BOUNDS["garment_chest_cm"][0], le=BOUNDS["garment_chest_cm"][1])
    fabric_stretch_pct: float = Field(ge=BOUNDS["fabric_stretch_pct"][0], le=BOUNDS["fabric_stretch_pct"][1])
    product_type_id: Literal[0, 1, 2]  # 0: Tops, 1: Bottoms, 2: Jackets
    # Which model to use; defaults to the baseline to keep the original API behaviour.
    model: Optional[Literal["baseline", "optimized"]] = "baseline"


@app.get("/", include_in_schema=False)
def web_ui():
    """Serve the fit-check web interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": baseline_model is not None,
        "expected_features_count": len(baseline_features),
        "available_models": list(MODELS),
    }


@app.get("/models")
def list_models():
    """Models the UI can choose from, which file each one is, plus the valid input ranges."""
    return {"available_models": list(MODELS), "models": MODEL_INFO, "bounds": BOUNDS, "classes": CLASS_LABELS}


@app.post("/reload")
def reload_models():
    """Re-read the model files from Models/ (called by the Airflow retrain DAG after a promotion)."""
    load_models()
    return {"available_models": list(MODELS), "models": MODEL_INFO}


@app.post("/predict")
def predict(data: PredictionInput):
    # Take one consistent snapshot, in case /reload swaps the models mid-request
    models, model_info = MODELS, MODEL_INFO
    model = models.get(data.model)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{data.model}' is not loaded")

    # 1. Raw input row (same columns for both models)
    raw = pd.DataFrame([data.model_dump(include=set(RAW_FEATURES))])[RAW_FEATURES]

    # 2. Baseline needs the hand-built one-hot schema; the optimized pipeline transforms internally
    model_input = build_baseline_features(raw, baseline_features) if data.model == "baseline" else raw

    # 3. Predict class & probability of every class (ordered by model.classes_)
    t0 = time.perf_counter()
    probabilities = model.predict_proba(model_input)[0]
    inference_ms = (time.perf_counter() - t0) * 1000
    class_ids = [int(c) for c in model.classes_]
    best = int(probabilities.argmax())
    prediction_id = class_ids[best]
    probs = {CLASS_LABELS.get(cid, str(cid)): round(float(p), 4) for cid, p in zip(class_ids, probabilities)}

    # 4. Audit trail: the exact row the model received and which model file answered
    model_input_row = {k: (v.item() if hasattr(v, "item") else v) for k, v in model_input.iloc[0].items()}
    log.info("[fitcheck] predict model=%s sha=%s input=%s -> %s %s (%.1f ms)",
             data.model, model_info[data.model]["sha256"], model_input_row,
             CLASS_LABELS.get(prediction_id), probs, inference_ms)

    return {
        "prediction_class": CLASS_LABELS.get(prediction_id, "Unknown"),
        "confidence": round(float(probabilities[best]), 4),
        "raw_class_id": prediction_id,
        "imc_index": round(float(bmi(data.height_cm, data.weight_kg)), 2),
        "model": data.model,
        "probabilities": probs,
        # Traceability fields (added; the original fields above are unchanged)
        "model_info": model_info[data.model],
        "model_input": model_input_row,
        "inference_ms": round(inference_ms, 2),
    }


if __name__ == "__main__":
    # Allows `python api/app.py` as well as the uvicorn CMD used in the Dockerfile.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "3000")))
