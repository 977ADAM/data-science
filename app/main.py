# app/main.py

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import json
import os

from app.schemas import PredictRequest, PredictResponse
from src.config import resolve_model_path
from src.versioning import sha256_file

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = resolve_model_path()


MODEL_DIR = PIPELINE_PATH.parent
# manifest лежит рядом с моделью, если это versioned-артефакт; для legacy его может не быть
_candidate_manifest = MODEL_DIR / "manifest.json"
MANIFEST_PATH = _candidate_manifest if _candidate_manifest.exists() else None
REQUIRE_MANIFEST = os.getenv("REQUIRE_MANIFEST", "0").strip().lower() in ("1", "true", "yes", "y")

@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = None
    try:
        pipe = joblib.load(str(PIPELINE_PATH))
    except Exception:
        logger.exception("Failed to load pipeline from %s", PIPELINE_PATH)
        pipe = None

    # manifest optional (но очень желателен)
    manifest = None
    try:
        if MANIFEST_PATH is not None and MANIFEST_PATH.exists():
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load manifest from %s", MANIFEST_PATH)
        manifest = None

    # Fail-closed режим для прода: если требуем manifest, а его нет — не поднимаем модель.
    if pipe is not None and REQUIRE_MANIFEST and manifest is None:
        logger.error(
            "REQUIRE_MANIFEST enabled but manifest is missing. Refusing to serve predictions. path=%s",
            PIPELINE_PATH,
        )
        pipe = None

    # Safety check: если есть manifest, проверяем целостность артефакта модели.
    # Несовпадение sha256 => не поднимаем модель (лучше fail-safe, чем тихий дрейф/подмена).
    if pipe is not None and manifest is not None:
        try:
            expected_sha = (((manifest.get("artifacts") or {}).get("pipeline_sha256")) or "").strip()
            if expected_sha:
                actual_sha = sha256_file(PIPELINE_PATH)
                if actual_sha != expected_sha:
                    logger.error(
                        "Pipeline sha256 mismatch! path=%s expected=%s actual=%s. Refusing to serve predictions.",
                        PIPELINE_PATH, expected_sha, actual_sha
                    )
                    pipe = None
        except Exception:
            logger.exception("Failed during pipeline integrity check for %s", PIPELINE_PATH)
            pipe = None

    app.state.pipe = pipe
    app.state.manifest = manifest
    yield

app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    pipe = getattr(app.state, "pipe", None)
    manifest = getattr(app.state, "manifest", None)
    return {
        "status": "ok" if pipe is not None else "pipeline_not_loaded",
        "pipeline_path": str(PIPELINE_PATH),
        "artifact_dir": str(MODEL_DIR),
        "pipeline_loaded": pipe is not None,
        "manifest_loaded": manifest is not None,
        "model_version": (manifest or {}).get("model_version"),
        "data_sha256": ((manifest or {}).get("data") or {}).get("raw_data_sha256"),
        "code_sha256": ((manifest or {}).get("code") or {}).get("code_sha256"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipe = getattr(app.state, "pipe", None)
    if pipe is None:
        logger.error("Pipeline is not loaded, cannot serve /predict")
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    # pydantic v2: model_dump; pydantic v1: dict
    customer_obj = req.customer
    payload = customer_obj.model_dump() if hasattr(customer_obj, "model_dump") else customer_obj.dict()
    # строго один клиент
    df = pd.DataFrame([payload])
    proba = float(pipe.predict_proba(df)[:, 1].item())
    return PredictResponse(churn_probability=proba)
