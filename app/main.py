# app/main.py

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import json
import os
import sys

from app.schemas import (
    PredictRequest,
    PredictResponse,
    DriftRequest,
    DriftResponse,
    UpliftRequest,
    UpliftResponse,
    ABSelectRequest,
    ABSelectResponse,
)
from src.config import resolve_model_path, resolve_uplift_model_path
from src.versioning import sha256_file
from src.drift import build_feature_frame, compare_to_reference, ks_stat_and_pvalue, psi_from_probs
from src.calibration import HoldoutCalibratedClassifier
from src.train_uplift import SklearnPipelineFactory
import numpy as np
import math
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock


logger = logging.getLogger(__name__)

PIPELINE_PATH = resolve_model_path()

UPLIFT_PIPELINE_PATH = resolve_uplift_model_path()

PSI_WARNING = 0.2
PSI_CRITICAL = 0.3

_metrics_lock = Lock()


def _asdict(obj) -> dict:
    """pydantic v2 -> model_dump; pydantic v1 -> dict; plain dict passthrough."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported payload type: {type(obj)!r}")


def _init_metrics_state(app: FastAPI) -> None:
    """
    In-memory metrics state.
    Notes:
    - This is process-local (resets on restart, not shared across replicas).
    - Good enough for demo / local monitoring; for prod consider Prometheus client multiprocess or external aggregation.
    """
    app.state.metrics = {
        "proba_sum": 0.0,
        "proba_count": 0,
        # last drift snapshot
        "psi_max": None,
        "unseen_pct": None,
        "psi_status": "ok",  # ok|warning|critical
    }

def _psi_status(psi_max: float | None) -> str:
    if psi_max is None:
        return "ok"
    if psi_max > PSI_CRITICAL:
        return "critical"
    if psi_max > PSI_WARNING:
        return "warning"
    return "ok"

def _format_prom_metric(name: str, value: float | int, labels: dict[str, str] | None = None) -> str:
    if labels:
        # stable ordering for readability
        items = ",".join([f'{k}="{str(v)}"' for k, v in sorted(labels.items())])
        return f"{name}{{{items}}} {value}\n"
    return f"{name} {value}\n"


MODEL_DIR = PIPELINE_PATH.parent
# manifest лежит рядом с моделью, если это versioned-артефакт; для legacy его может не быть
_candidate_manifest = MODEL_DIR / "manifest.json"
MANIFEST_PATH = _candidate_manifest if _candidate_manifest.exists() else None
REQUIRE_MANIFEST = os.getenv("REQUIRE_MANIFEST", "0").strip().lower() in ("1", "true", "yes", "y")

@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_metrics_state(app)
    pipe = None
    uplift_pipe = None
    try:
        # ---------------------------------------------------------------------
        # Backward-compat for old pickles that were created when the class
        # was defined under multiprocessing main module name "__mp_main__".
        #
        # When unpickling, Python does:
        #   sys.modules["__mp_main__"].HoldoutCalibratedClassifier
        # In uvicorn, "__mp_main__" points to uvicorn module => AttributeError.
        # We inject the symbol so old artifacts load.
        # Future artifacts will be stable because the class now lives in src.calibration.
        # ---------------------------------------------------------------------
        mp_main = sys.modules.get("__mp_main__")
        if mp_main is not None:
            if not hasattr(mp_main, "HoldoutCalibratedClassifier"):
                setattr(mp_main, "HoldoutCalibratedClassifier", HoldoutCalibratedClassifier)
            if not hasattr(mp_main, "SklearnPipelineFactory"):
                setattr(mp_main, "SklearnPipelineFactory", SklearnPipelineFactory)
        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            if not hasattr(main_mod, "HoldoutCalibratedClassifier"):
                setattr(main_mod, "HoldoutCalibratedClassifier", HoldoutCalibratedClassifier)
            if not hasattr(main_mod, "SklearnPipelineFactory"):
                setattr(main_mod, "SklearnPipelineFactory", SklearnPipelineFactory)

        pipe = joblib.load(str(PIPELINE_PATH))
    except Exception:
        logger.exception("Failed to load pipeline from %s", PIPELINE_PATH)
        pipe = None


    # Optional: uplift T-learner (two-model). Not required for churn /predict.
    try:
        if UPLIFT_PIPELINE_PATH.exists():
            uplift_pipe = joblib.load(str(UPLIFT_PIPELINE_PATH))
    except Exception:
        logger.exception("Failed to load uplift model from %s", UPLIFT_PIPELINE_PATH)
        uplift_pipe = None

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
    app.state.uplift_pipe = uplift_pipe
    app.state.manifest = manifest
    yield

app = FastAPI(title="Churn Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    pipe = getattr(app.state, "pipe", None)
    uplift_pipe = getattr(app.state, "uplift_pipe", None)
    manifest = getattr(app.state, "manifest", None)
    return {
        "status": "ok" if pipe is not None else "pipeline_not_loaded",
        "pipeline_path": str(PIPELINE_PATH),
        "uplift_pipeline_path": str(UPLIFT_PIPELINE_PATH),
        "artifact_dir": str(MODEL_DIR),
        "pipeline_loaded": pipe is not None,
        "uplift_pipeline_loaded": uplift_pipe is not None,
        "manifest_loaded": manifest is not None,
        "model_version": (manifest or {}).get("model_version"),
        "data_sha256": ((manifest or {}).get("data") or {}).get("raw_data_sha256"),
        "code_sha256": ((manifest or {}).get("code") or {}).get("code_sha256"),
    }



@app.post("/uplift", response_model=UpliftResponse)
def uplift(req: UpliftRequest):
    """
    Two-model uplift (T-learner) scoring.

    Returns:
      - p_treated  = P(Y=1|X, do(treat=1))
      - p_control  = P(Y=1|X, do(treat=0))
      - uplift     = p_treated - p_control
    """
    uplift_pipe = getattr(app.state, "uplift_pipe", None)
    if uplift_pipe is None:
        raise HTTPException(status_code=500, detail="Uplift model not loaded")

    customer_obj = req.customer
    payload = _asdict(customer_obj)
    df = pd.DataFrame([payload])

    try:
        p_t = float(uplift_pipe.predict_proba_treated(df).reshape(-1)[0])
        p_c = float(uplift_pipe.predict_proba_control(df).reshape(-1)[0])
        u = float(p_t - p_c)
    except Exception:
        logger.exception("Failed to compute uplift")
        raise HTTPException(status_code=500, detail="Failed to compute uplift")

    return UpliftResponse(p_treated=p_t, p_control=p_c, uplift=u)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    pipe = getattr(app.state, "pipe", None)
    if pipe is None:
        logger.error("Pipeline is not loaded, cannot serve /predict")
        raise HTTPException(status_code=500, detail="Pipeline not loaded")

    customer_obj = req.customer
    payload = _asdict(customer_obj)
    # строго один клиент
    df = pd.DataFrame([payload])
    try:
        proba = float(pipe.predict_proba(df)[:, 1].item())
    except Exception:
        logger.exception("Failed to compute churn probability")
        raise HTTPException(status_code=500, detail="Failed to compute prediction")
    if not math.isfinite(proba):
        logger.error("Non-finite churn probability: %s", proba)
        raise HTTPException(status_code=500, detail="Non-finite prediction output")

    # update in-memory avg proba
    try:
        if math.isfinite(proba):
            with _metrics_lock:
                m = getattr(app.state, "metrics", None)
                if m is not None:
                    m["proba_sum"] = float(m.get("proba_sum", 0.0)) + float(proba)
                    m["proba_count"] = int(m.get("proba_count", 0)) + 1
    except Exception:
        # metrics must never break prediction endpoint
        pass

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
        rows.append(_asdict(c))
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

                    psi = psi_from_probs(ref_probs, cur_probs)

                    # KS: approximate reference distribution from bins (midpoints + ref_probs)
                    try:
                        mids = (edges[:-1] + edges[1:]) / 2.0
                        if mids.size == ref_probs.size and ref_probs.sum() > 0:
                            n_syn = int(min(5000, max(200, proba.size * 2)))
                            syn = np.random.default_rng(42).choice(mids, size=n_syn, p=ref_probs / ref_probs.sum())
                            ks_stat, ks_pvalue = ks_stat_and_pvalue(syn, proba)
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
        logger.exception("Failed to compute prediction drift")
        pred_metrics = {"error": "failed_to_compute_prediction_drift"}

    out = {
        "data_drift": drift_metrics,
        "prediction_drift": pred_metrics,
        "model_version": (manifest or {}).get("model_version"),
    }

    # update /metrics snapshot: psi_max + unseen%
    try:
        psi_vals = []
        unseen_vals = []

        # numeric psi
        for _, v in (drift_metrics.get("numeric") or {}).items():
            psi = v.get("psi") if isinstance(v, dict) else None
            if psi is not None and isinstance(psi, (int, float)) and math.isfinite(float(psi)):
                psi_vals.append(float(psi))

        # categorical psi + unseen_mass
        for _, v in (drift_metrics.get("categorical") or {}).items():
            if not isinstance(v, dict):
                continue
            psi = v.get("psi")
            if psi is not None and isinstance(psi, (int, float)) and math.isfinite(float(psi)):
                psi_vals.append(float(psi))
            um = v.get("unseen_mass")
            if um is not None and isinstance(um, (int, float)) and math.isfinite(float(um)):
                unseen_vals.append(float(um))

        psi_max = max(psi_vals) if psi_vals else None
        unseen_pct = (max(unseen_vals) * 100.0) if unseen_vals else None

        with _metrics_lock:
            m = getattr(app.state, "metrics", None)
            if m is not None:
                m["psi_max"] = psi_max
                m["unseen_pct"] = unseen_pct
                m["psi_status"] = _psi_status(psi_max)
    except Exception:
        pass

    return DriftResponse(drift=out)



@app.get("/metrics")
def metrics():
    """
    Prometheus-friendly text exposition.
    Exposes:
      - churn_avg_proba
      - churn_psi_max (last /drift snapshot)
      - churn_unseen_categories_pct (last /drift snapshot, max over categorical features)
      - churn_psi_status{status="ok|warning|critical"} (one-hot)
    Thresholds:
      PSI > 0.2 -> warning
      PSI > 0.3 -> critical
    """
    with _metrics_lock:
        m = getattr(app.state, "metrics", None) or {}
        proba_sum = float(m.get("proba_sum", 0.0))
        proba_count = int(m.get("proba_count", 0))
        avg_proba = (proba_sum / proba_count) if proba_count > 0 else float("nan")
        psi_max = m.get("psi_max", None)
        unseen_pct = m.get("unseen_pct", None)
        status = str(m.get("psi_status", "ok"))

    lines = []
    # help/type (nice to have)
    lines.append("# HELP churn_avg_proba Average predicted churn probability since process start.\n")
    lines.append("# TYPE churn_avg_proba gauge\n")
    lines.append(_format_prom_metric("churn_avg_proba", avg_proba))

    lines.append("# HELP churn_psi_max Max PSI over features from the last /drift call.\n")
    lines.append("# TYPE churn_psi_max gauge\n")
    lines.append(_format_prom_metric("churn_psi_max", float(psi_max) if psi_max is not None else float("nan")))

    lines.append("# HELP churn_unseen_categories_pct Max share (%%) of unseen categories from the last /drift call.\n")
    lines.append("# TYPE churn_unseen_categories_pct gauge\n")
    lines.append(_format_prom_metric("churn_unseen_categories_pct", float(unseen_pct) if unseen_pct is not None else float("nan")))

    lines.append("# HELP churn_psi_status PSI status derived from churn_psi_max thresholds (ok/warning/critical).\n")
    lines.append("# TYPE churn_psi_status gauge\n")
    for s in ("ok", "warning", "critical"):
        lines.append(_format_prom_metric("churn_psi_status", 1 if status == s else 0, labels={"status": s}))

    body = "".join(lines)
    return Response(content=body, media_type="text/plain; version=0.0.4")

@app.post("/ab/select", response_model=ABSelectResponse)
def ab_select(req: ABSelectRequest):
    """
    A/B targeting:
      - Control  : TOP-K by churn probability
      - Treatment: TOP-K by uplift score
    """
    pipe = getattr(app.state, "pipe", None)
    uplift_pipe = getattr(app.state, "uplift_pipe", None)

    if pipe is None:
        raise HTTPException(status_code=500, detail="Churn model not loaded")
    if uplift_pipe is None:
        raise HTTPException(status_code=500, detail="Uplift model not loaded")

    customers = req.customers or []
    k = int(req.k)

    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be > 0")
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="customers must be non-empty")

    rows = []
    for c in customers:
        rows.append(_asdict(c))
    df = pd.DataFrame(rows)

    # -------- Control: churn score --------
    try:
        churn_scores = pipe.predict_proba(df)[:, 1]
    except Exception:
        logger.exception("Failed to compute churn scores for A/B selection")
        raise HTTPException(status_code=500, detail="Failed to score churn probabilities")
    churn_scores = np.asarray(churn_scores, dtype=float)

    order_churn = np.argsort(-churn_scores)
    control_top_k_idx = order_churn[: min(k, len(order_churn))].tolist()

    # -------- Treatment: uplift score --------
    try:
        uplift_scores = uplift_pipe.predict_uplift(df)
    except Exception:
        logger.exception("Failed to compute uplift scores for A/B selection")
        raise HTTPException(status_code=500, detail="Failed to score uplift")
    uplift_scores = np.asarray(uplift_scores, dtype=float)

    order_uplift = np.argsort(-uplift_scores)
    treatment_top_k_idx = order_uplift[: min(k, len(order_uplift))].tolist()

    return ABSelectResponse(
        control_top_k_idx=control_top_k_idx,
        treatment_top_k_idx=treatment_top_k_idx,
    )