# app/main.py

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import json
import os

from app.schemas import PredictRequest, PredictResponse, DriftRequest, DriftResponse
from src.config import resolve_model_path
from src.versioning import sha256_file
from src.drift import build_feature_frame, compare_to_reference
import numpy as np
import math

def _ks_stat_and_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Two-sample KS test with asymptotic p-value approximation (no scipy dependency)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n1 = x.size
    n2 = y.size
    if n1 < 2 or n2 < 2:
        return None, None

    x_sort = np.sort(x)
    y_sort = np.sort(y)
    data_all = np.sort(np.unique(np.concatenate([x_sort, y_sort])))
    cdf1 = np.searchsorted(x_sort, data_all, side="right") / n1
    cdf2 = np.searchsorted(y_sort, data_all, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic p-value approximation via Kolmogorov distribution.
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    s = 0.0
    for j in range(1, 101):
        term = (-1) ** (j - 1) * math.exp(-2.0 * (j * lam) ** 2)
        s += term
        if abs(term) < 1e-10:
            break
    p = max(0.0, min(1.0, 2.0 * s))
    return d, p


def _psi(ref_p: np.ndarray, cur_p: np.ndarray, eps: float = 1e-6) -> float | None:
    ref_p = np.asarray(ref_p, dtype=float)
    cur_p = np.asarray(cur_p, dtype=float)
    if ref_p.size == 0 or cur_p.size == 0 or ref_p.size != cur_p.size:
        return None
    ref_p = np.clip(ref_p, eps, 1.0)
    cur_p = np.clip(cur_p, eps, 1.0)
    ref_p = ref_p / ref_p.sum()
    cur_p = cur_p / cur_p.sum()
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


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


@app.post("/drift", response_model=DriftResponse)
def drift(req: DriftRequest):
    """
    Compare incoming batch vs train reference (stored in manifest.json):
    - numeric: PSI + KS
    - categorical: frequency drift (L1 + PSI)
    - prediction drift: distribution of predicted probabilities
    """
    pipe = getattr(app.state, "pipe", None)
    manifest = getattr(app.state, "manifest", None)
    if pipe is None:
        logger.error("Pipeline is not loaded, cannot serve /drift")
        raise HTTPException(status_code=500, detail="Pipeline not loaded")
    if not manifest:
        raise HTTPException(status_code=500, detail="Manifest not loaded; drift reference unavailable")

    drift_ref = (((manifest.get("data") or {}).get("profile") or {}).get("drift_reference") or {})
    feat_ref = (drift_ref.get("features") or {})
    pred_ref = (drift_ref.get("prediction_proba") or {})
    if not feat_ref:
        raise HTTPException(status_code=500, detail="Drift reference missing in manifest")

    # batch -> df
    customers = req.customers or []
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="customers must be a non-empty list")
    rows = []
    for c in customers:
        rows.append(c.model_dump() if hasattr(c, "model_dump") else c.dict())
    df_raw = pd.DataFrame(rows)

    # feature-frame (align/clean/feat if possible)
    df_feat = build_feature_frame(pipe, df_raw)

    drift_metrics = compare_to_reference(df_feat, feat_ref)

    # prediction drift
    pred_metrics = {}
    try:
        proba = pipe.predict_proba(df_raw)[:, 1]
        proba = np.asarray(proba, dtype=float)
        proba = proba[np.isfinite(proba)]
        if proba.size:
            # PSI/KS based on reference edges (quantile bins) + (optional) ref_bin_probs.
            psi = None
            ks_stat = None
            ks_pvalue = None

            ref_edges = pred_ref.get("edges")
            ref_bin_probs = pred_ref.get("ref_bin_probs")
            if ref_edges and isinstance(ref_edges, list) and len(ref_edges) >= 3:
                edges = np.asarray(ref_edges, dtype=float)
                edges = np.unique(edges)
                if edges.size >= 3:
                    # current bin probs
                    idx = np.digitize(proba, edges[1:-1], right=False)
                    counts = np.bincount(idx, minlength=edges.size - 1).astype(float)
                    cur_probs = counts / counts.sum() if counts.sum() > 0 else np.zeros(edges.size - 1, dtype=float)

                    # prefer stored ref_bin_probs; fallback to uniform (quantile bins).
                    if ref_bin_probs and isinstance(ref_bin_probs, list) and len(ref_bin_probs) == int(edges.size - 1):
                        ref_probs = np.asarray(ref_bin_probs, dtype=float)
                    else:
                        ref_probs = np.ones(edges.size - 1, dtype=float) / float(edges.size - 1)

                    psi = _psi(ref_probs, cur_probs)

                    # KS: approximate reference distribution from bins (midpoints + ref_probs)
                    try:
                        mids = (edges[:-1] + edges[1:]) / 2.0
                        if mids.size == ref_probs.size and ref_probs.sum() > 0:
                            n_syn = int(min(5000, max(200, proba.size * 2)))
                            syn = np.random.default_rng(42).choice(mids, size=n_syn, p=ref_probs / ref_probs.sum())
                            ks_stat, ks_pvalue = _ks_stat_and_pvalue(syn, proba)
                    except Exception:
                        ks_stat, ks_pvalue = None, None

            pred_metrics = {
                "rows": int(proba.size),
                "ref_mean": pred_ref.get("mean"),
                "ref_std": pred_ref.get("std"),
                "cur_mean": float(np.mean(proba)),
                "cur_std": float(np.std(proba, ddof=0)),
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
            }
        else:
            pred_metrics = {"rows": 0, "psi": None}
    except Exception:
        pred_metrics = {"error": "failed_to_compute_prediction_drift"}

    out = {
        "data_drift": drift_metrics,
        "prediction_drift": pred_metrics,
        "model_version": (manifest or {}).get("model_version"),
    }
    return DriftResponse(drift=out)
