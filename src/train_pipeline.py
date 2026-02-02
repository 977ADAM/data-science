# src/train_pipeline.py

import joblib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.pipeline import make_pipeline
from src.drift import build_feature_frame, build_reference_profile
from src.config import (
    ARTIFACTS_DIR,
    RAW_DATA,
    TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    THRESHOLD,
)
from src.versioning import (
    Manifest,
    atomic_write_json,
    atomic_write_text,
    build_manifest,
    now_utc_compact,
    safe_mkdir,
    sha256_file,
)

def train_and_save():
    df = load_data()
    df = clean_data(df)

    # y
    y = df[TARGET].astype(int)

    # X raw (важно: оставляем сырые категориальные!)
    X = df.drop(columns=[TARGET])

    # --------- Data profile (quality snapshot) ---------
    # This is intentionally lightweight but very useful for debugging / drift detection.
    # Keep it stable across runs (sort keys; basic python types).
    missing_rate = (df.isna().mean()).to_dict()
    data_profile = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_rate": float(y.mean()) if len(y) else None,
        "missing_rate": {str(k): float(v) for k, v in missing_rate.items()},
        "drift_reference": {
            "features": ref_feature_profile,
            "prediction_proba": pred_ref,
        },
    }


    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipe = make_pipeline(X_train)

    # 1) обучаем базовый pipeline
    pipe.fit(X_train, y_train)

    # 2) калибруем вероятности на holdout
    # используем prefit, чтобы не переобучать pipeline
    pipe = CalibratedClassifierCV(
        base_estimator=pipe,
        method="isotonic",
        cv="prefit",
    )
    pipe.fit(X_test, y_test)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= THRESHOLD).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

    # Диагностика порога (боевой THRESHOLD не меняем)
    grid = [i / 100 for i in range(1, 100)]
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y_test, (proba >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"Best F1 threshold on holdout: {best_t:.2f} (F1={best_f1:.4f}); current THRESHOLD={THRESHOLD}")

    # --------- Drift reference snapshot (train distributions) ---------
    # Build feature-frame after align/clean/feat (categoricals preserved).
    try:
        X_train_feat = build_feature_frame(pipe, X_train)
        X_test_feat = build_feature_frame(pipe, X_test)
        ref_feature_profile = build_reference_profile(X_train_feat, numeric_bins=10, max_categories=50)
    except Exception:
        ref_feature_profile = {"error": "failed_to_build_feature_reference"}

    # Prediction drift reference: distribution of calibrated probabilities on holdout (stable + production-like).
    try:
        proba_ref = pipe.predict_proba(X_test)[:, 1]
        proba_ref = np.asarray(proba_ref, dtype=float)
        proba_ref = proba_ref[np.isfinite(proba_ref)]
        if proba_ref.size:
            qs = np.linspace(0.0, 1.0, 11)
            edges = np.quantile(proba_ref, qs)
            edges = np.unique(edges)
            # keep mean/std + edges (binning in API)
            pred_ref = {
                "mean": float(np.mean(proba_ref)),
                "std": float(np.std(proba_ref, ddof=0)),
                "edges": [float(x) for x in edges.tolist()] if edges.size >= 3 else None,
            }
        else:
            pred_ref = {"mean": None, "std": None, "edges": None}
    except Exception:
        pred_ref = {"error": "failed_to_build_prediction_reference"}


    # --------- Versioned artifacts ---------
    repo_root = Path(__file__).resolve().parents[1]
    data_sha = sha256_file(RAW_DATA) if Path(RAW_DATA).exists() else "no_data"
    code_hint = sha256_file(repo_root / "pyproject.toml") if (repo_root / "pyproject.toml").exists() else "no_pyproject"
    # компактная версия: timestamp + 6 символов от data/code
    model_version = f"{now_utc_compact()}_{data_sha[:6]}{code_hint[:6]}"

    model_dir = safe_mkdir(ARTIFACTS_DIR / model_version)
    model_path = model_dir / "churn_pipeline.pkl"
    manifest_path = model_dir / "manifest.json"

    # Пишем модель атомарно (через временный файл)
    tmp_model_path = model_path.with_suffix(".pkl.tmp")
    joblib.dump(pipe, tmp_model_path)
    os.replace(tmp_model_path, model_path)

    artifacts = {
        "pipeline_path": str(model_path),
        "pipeline_sha256": sha256_file(model_path),
        "manifest_path": str(manifest_path),
    }
    training_params = {
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": THRESHOLD,
        "train_size": int(len(X_train)),
        "test_size_rows": int(len(X_test)),
    }
    manifest: Manifest = build_manifest(
        model_version=model_version,
        repo_root=repo_root,
        raw_data_path=Path(RAW_DATA),
        training_params=training_params,
        artifacts=artifacts,
        data_profile=data_profile,
    )
    atomic_write_json(manifest_path, manifest.to_dict())

    # Обновляем latest (symlink если можно) + latest.txt (portable)
    safe_mkdir(ARTIFACTS_DIR)
    latest_link = ARTIFACTS_DIR / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(model_dir, target_is_directory=True)
    except Exception:
        # если symlink запрещён, просто не падаем
        pass
    atomic_write_text(ARTIFACTS_DIR / "latest.txt", model_version + "\n")

    print(f"Saved model: {model_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Model version: {model_version}")


if __name__ == "__main__":
    train_and_save()
