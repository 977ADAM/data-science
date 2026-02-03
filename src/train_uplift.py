# src/train_uplift.py

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.calibration import HoldoutCalibratedClassifier
from src.config import ARTIFACTS_DIR, RAW_DATA, RANDOM_STATE, TEST_SIZE, TARGET
from src.data_loader import load_data
from src.pipeline import make_pipeline
from src.preprocessing import clean_data
from src.pipeline import make_regression_pipeline
from src.uplift import DRLearner, SLearner, TLearner, XLearner
from src.versioning import (
    Manifest,
    atomic_write_json,
    atomic_write_text,
    build_manifest,
    now_utc_compact,
    safe_mkdir,
    sha256_file,
)


def train_and_save_uplift():
    """
    Train two-model uplift (T-learner) for binary outcome.

    Requirements:
      - dataset has outcome column TARGET (default: Churn)
      - dataset has binary treatment column (env TREATMENT_COL, default: treatment)
        where 1 = treated, 0 = control.
    """

    TREATMENT_COL = os.getenv("TREATMENT_COL", "treatment").strip()
    UPLIFT_LEARNER = os.getenv("UPLIFT_LEARNER", "t").strip().lower()
    # accepted: t | s | x | dr

    df = load_data()
    df = clean_data(df)

    if TREATMENT_COL not in df.columns:
        raise ValueError(
            f"Treatment column {TREATMENT_COL!r} is missing in data. "
            "Provide it or set TREATMENT_COL env var."
        )

    # outcome
    y = df[TARGET].astype(int).to_numpy(dtype=int)

    # treatment
    t = df[TREATMENT_COL].astype(int).to_numpy(dtype=int)
    if not set(np.unique(t)).issubset({0, 1}):
        raise ValueError(f"Treatment column {TREATMENT_COL!r} must be binary 0/1")

    # raw X
    X = df.drop(columns=[TARGET, TREATMENT_COL])

    # split (keep treatment mix stable)
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X,
        t,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=t,
    )

    # Keep as DataFrames for pipeline column handling
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    t_train = np.asarray(t_train, dtype=int)
    t_test = np.asarray(t_test, dtype=int)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    # Factories
    def clf_factory():
        # IMPORTANT: build a fresh pipeline each time
        return make_pipeline(X_train)

    def reg_factory():
        return make_regression_pipeline(X_train)

    # Learner selection
    if UPLIFT_LEARNER in ("t", "t-learner", "tlearner"):
        learner = TLearner(clf_factory)
        learner.fit(X_train, t_train, y_train)
    elif UPLIFT_LEARNER in ("s", "s-learner", "slearner"):
        # For S-learner we must include the treatment feature as an input column
        X_train_s = X_train.copy()
        X_train_s[TREATMENT_COL] = t_train

        def s_factory():
            return make_pipeline(X_train_s)

        learner = SLearner(s_factory, treatment_col=TREATMENT_COL)
        learner.fit(X_train, t_train, y_train)
    elif UPLIFT_LEARNER in ("x", "x-learner", "xlearner"):
        learner = XLearner(
            outcome_model_factory=clf_factory,
            effect_model_factory=reg_factory,
            propensity_model_factory=clf_factory,
        )
        learner.fit(X_train, t_train, y_train)
    elif UPLIFT_LEARNER in ("dr", "dr-learner", "drlearner", "doubly-robust", "doubly_robust"):
        learner = DRLearner(
            outcome_model_factory=clf_factory,
            propensity_model_factory=clf_factory,
            effect_model_factory=reg_factory,
        )
        learner.fit(X_train, t_train, y_train)
    else:
        raise ValueError(
            f"Unknown UPLIFT_LEARNER={UPLIFT_LEARNER!r}. Expected one of: t, s, x, dr."
        )

    # Optional: calibrate each model on its group holdout (if enough data)
    try:
        # Only applies to learners that expose per-group outcome models (TLearner)
        mt = getattr(getattr(learner, "artifacts_", None), "model_treated", None)
        mc = getattr(getattr(learner, "artifacts_", None), "model_control", None)
        if mt is None or mc is None:
            raise AttributeError("no per-group outcome models to calibrate")

        mask_tt = t_test == 1
        mask_tc = t_test == 0

        if int(mask_tt.sum()) >= 50 and int(mask_tc.sum()) >= 50:
            mt_cal = HoldoutCalibratedClassifier(mt, method="isotonic").fit(X_test[mask_tt], y_test[mask_tt])
            mc_cal = HoldoutCalibratedClassifier(mc, method="isotonic").fit(X_test[mask_tc], y_test[mask_tc])
            learner.artifacts_.model_treated = mt_cal
            learner.artifacts_.model_control = mc_cal
    except Exception:
        # calibration is a nice-to-have; never fail the training run because of it
        pass

    # Group quality metrics (classification AUC per group)
    metrics = {}
    try:
        mask_tt = t_test == 1
        mask_tc = t_test == 0
        if int(mask_tt.sum()) >= 2 and len(np.unique(y_test[mask_tt])) == 2:
            auc_t = float(roc_auc_score(y_test[mask_tt], learner.predict_proba_treated(X_test[mask_tt])))
        else:
            auc_t = None
        if int(mask_tc.sum()) >= 2 and len(np.unique(y_test[mask_tc])) == 2:
            auc_c = float(roc_auc_score(y_test[mask_tc], learner.predict_proba_control(X_test[mask_tc])))
        else:
            auc_c = None
        metrics = {
            "treated_rows": int(mask_tt.sum()),
            "control_rows": int(mask_tc.sum()),
            "auc_treated": auc_t,
            "auc_control": auc_c,
            "uplift_learner": UPLIFT_LEARNER,
        }
    except Exception:
        metrics = {"error": "failed_to_compute_group_auc"}

    # --------- Versioned artifacts ---------
    repo_root = Path(__file__).resolve().parents[1]
    data_sha = sha256_file(RAW_DATA) if Path(RAW_DATA).exists() else "no_data"
    code_hint = sha256_file(repo_root / "pyproject.toml") if (repo_root / "pyproject.toml").exists() else "no_pyproject"
    model_version = f"{now_utc_compact()}_{data_sha[:6]}{code_hint[:6]}"

    model_dir = safe_mkdir(ARTIFACTS_DIR / model_version)
    model_path = model_dir / "uplift_tlearner.pkl"
    manifest_path = model_dir / "manifest_uplift.json"

    tmp_model_path = model_path.with_suffix(".pkl.tmp")
    joblib.dump(learner, tmp_model_path)
    os.replace(tmp_model_path, model_path)

    artifacts = {
        "uplift_pipeline_path": str(model_path),
        "uplift_pipeline_sha256": sha256_file(model_path),
        "manifest_path": str(manifest_path),
    }
    training_params = {
        "target": TARGET,
        "treatment_col": TREATMENT_COL,
        "uplift_learner": UPLIFT_LEARNER,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "metrics": metrics,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    manifest: Manifest = build_manifest(
        model_version=model_version,
        repo_root=repo_root,
        raw_data_path=Path(RAW_DATA),
        training_params=training_params,
        artifacts=artifacts,
        data_profile=None,
    )
    atomic_write_json(manifest_path, manifest.to_dict())

    # Latest pointers (portable)
    safe_mkdir(ARTIFACTS_DIR)
    latest_link = ARTIFACTS_DIR / "latest_uplift"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(model_dir, target_is_directory=True)
    except Exception:
        # если symlink запрещён, просто не падаем
        pass
    atomic_write_text(ARTIFACTS_DIR / "latest_uplift.txt", model_version + "\n")

    print(f"Saved uplift model: {model_path}")
    print(f"Saved uplift manifest: {manifest_path}")
    print(f"Model version: {model_version}")
    print("Group metrics:", metrics)


if __name__ == "__main__":
    train_and_save_uplift()
