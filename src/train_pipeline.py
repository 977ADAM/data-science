# src/train_pipeline.py

import joblib
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from src.data_loader import load_data
from src.preprocessing import clean_data
from src.pipeline import make_pipeline
from src.drift import build_feature_frame, build_reference_profile
from src.calibration import HoldoutCalibratedClassifier
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

def _expected_value(
    p_churn: np.ndarray,
    *,
    ltv_saved: float,
    cost_action: float,
    retention_uplift: float = 1.0,
) -> np.ndarray:
    """
    EV per customer:
      EV = p_churn * LTV_saved * retention_uplift - Cost_action
    retention_uplift: вероятность, что action действительно предотвращает churn (0..1).
    """
    p = np.asarray(p_churn, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    ru = float(np.clip(retention_uplift, 0.0, 1.0))
    return p * float(ltv_saved) * ru - float(cost_action)

def _ev_at_k(
    proba: np.ndarray,
    *,
    k: int,
    ltv_saved: float,
    cost_action: float,
    retention_uplift: float = 1.0,
) -> dict:
    """
    EV@K: ожидаемая выгода, если таргетировать TOP-K клиентов по p_churn.
    """
    p = np.asarray(proba, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return {"k": int(k), "ev_total": 0.0, "ev_per_customer": 0.0, "cutoff": None}

    k = int(max(0, min(int(k), int(p.size))))
    if k == 0:
        return {"k": 0, "ev_total": 0.0, "ev_per_customer": 0.0, "cutoff": None}

    order = np.argsort(-p)  # desc
    top_idx = order[:k]
    top_p = p[top_idx]
    ev = _expected_value(top_p, ltv_saved=ltv_saved, cost_action=cost_action, retention_uplift=retention_uplift)
    ev_total = float(np.sum(ev))
    cutoff = float(np.min(top_p))  # порог, который соответствует TOP-K
    return {
        "k": int(k),
        "ev_total": ev_total,
        "ev_per_customer": float(ev_total / k) if k else 0.0,
        "cutoff": cutoff,
    }

def _find_optimal_cutoff_by_ev(
    proba: np.ndarray,
    *,
    ltv_saved: float,
    cost_action: float,
    retention_uplift: float = 1.0,
    grid: np.ndarray | None = None,
) -> dict:
    """
    Выбираем порог не по F1, а по максимуму суммарного EV (на holdout).
    Возвращает: best_threshold, best_ev_total, targeted_count, expected_uplift (== best_ev_total)
    """
    p = np.asarray(proba, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return {"best_threshold": 0.5, "best_ev_total": 0.0, "targeted_count": 0, "expected_uplift": 0.0}

    if grid is None:
        # достаточно плотная сетка, но без крайних 0/1
        grid = np.linspace(0.01, 0.99, 99)

    best_t = 0.5
    best_ev = -float("inf")
    best_cnt = 0

    for t in grid:
        mask = p >= float(t)
        if not np.any(mask):
            ev_total = 0.0
            cnt = 0
        else:
            ev_total = float(
                np.sum(
                    _expected_value(
                        p[mask],
                        ltv_saved=ltv_saved,
                        cost_action=cost_action,
                        retention_uplift=retention_uplift,
                    )
                )
            )
            cnt = int(np.sum(mask))
        if ev_total > best_ev:
            best_ev = ev_total
            best_t = float(t)
            best_cnt = cnt

    return {
        "best_threshold": float(best_t),
        "best_ev_total": float(best_ev),
        "targeted_count": int(best_cnt),
        # baseline "do nothing" = 0, поэтому uplift == EV
        "expected_uplift": float(best_ev),
    }

def _simulate_retention_campaign(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    k: int,
    ltv_saved: float,
    cost_action: float,
    retention_uplift: float = 1.0,
    n_sim: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Симуляция retention-кампании (эмпирическая):
    - таргетируем TOP-K по proba
    - если клиент реально churn (y=1), то спасаем LTV с вероятностью retention_uplift
    - если y=0, платим cost_action впустую
    Возвращает распределение прибыли по симуляциям.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    ok = np.isfinite(p)
    y = y[ok]
    p = p[ok]
    n = int(p.size)
    if n == 0:
        return {"k": int(k), "n": 0, "profit_mean": 0.0, "profit_std": 0.0}

    k = int(max(0, min(int(k), n)))
    if k == 0:
        return {"k": 0, "n": n, "profit_mean": 0.0, "profit_std": 0.0}

    order = np.argsort(-p)
    idx = order[:k]
    yk = y[idx]

    rng = np.random.default_rng(int(random_state))
    ru = float(np.clip(retention_uplift, 0.0, 1.0))
    profits = []
    for _ in range(int(max(1, n_sim))):
        saved = (yk == 1) & (rng.random(k) < ru)
        profit = float(np.sum(saved) * float(ltv_saved) - k * float(cost_action))
        profits.append(profit)
    profits = np.asarray(profits, dtype=float)
    return {
        "k": int(k),
        "n": int(n),
        "profit_mean": float(np.mean(profits)),
        "profit_std": float(np.std(profits, ddof=0)),
    }

def train_and_save():
    df = load_data()
    df = clean_data(df)

    # y
    y = df[TARGET].astype(int)

    # X raw (важно: оставляем сырые категориальные!)
    X = df.drop(columns=[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    base_pipe = make_pipeline(X_train)

    # 1) обучаем базовый pipeline
    base_pipe.fit(X_train, y_train)

    # 2) калибруем вероятности на holdout
    # Не используем CalibratedClassifierCV, чтобы не упираться в classes_/cross_val_predict.
    pipe = HoldoutCalibratedClassifier(
        estimator=base_pipe,
        method="isotonic",   # либо "sigmoid"
    ).fit(X_test, y_test)

    proba = pipe.predict_proba(X_test)[:, 1]
    # ---------------- Business / EV simulation ----------------
    # Эти параметры можно вынести в config/env позже; сейчас задаём устойчивые дефолты.
    LTV_SAVED = float(os.getenv("LTV_SAVED", "1000"))          # условная "спасённая" ценность клиента
    COST_ACTION = float(os.getenv("COST_ACTION", "50"))        # стоимость звонка/скидки/оффера
    RETENTION_UPLIFT = float(os.getenv("RETENTION_UPLIFT", "1.0"))  # эффективность удержания (0..1)

    # optimal cutoff по max EV (на holdout)
    opt = _find_optimal_cutoff_by_ev(
        proba,
        ltv_saved=LTV_SAVED,
        cost_action=COST_ACTION,
        retention_uplift=RETENTION_UPLIFT,
    )
    selected_threshold = float(opt["best_threshold"])
    expected_uplift = float(opt["expected_uplift"])

    preds = (proba >= selected_threshold).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

    # EV@K (ожидаемая) + симуляция (эмпирическая по y_test)
    n_test = int(len(y_test))
    k_grid = [
        max(1, int(0.01 * n_test)),
        max(1, int(0.05 * n_test)),
        max(1, int(0.10 * n_test)),
    ]
    ev_at_k = [
        _ev_at_k(proba, k=k, ltv_saved=LTV_SAVED, cost_action=COST_ACTION, retention_uplift=RETENTION_UPLIFT)
        for k in k_grid
    ]
    sim_at_k = [
        _simulate_retention_campaign(
            y_test,
            proba,
            k=k,
            ltv_saved=LTV_SAVED,
            cost_action=COST_ACTION,
            retention_uplift=RETENTION_UPLIFT,
            n_sim=int(os.getenv("RETENTION_SIM_N", "200")),
            random_state=RANDOM_STATE,
        )
        for k in k_grid
    ]

    print(
        f"Selected threshold by max EV (holdout): {selected_threshold:.2f} "
        f"| expected_uplift={expected_uplift:.2f} "
        f"| targeted={int(opt['targeted_count'])}/{n_test}"
    )
    print(f"Business params: LTV_SAVED={LTV_SAVED}, COST_ACTION={COST_ACTION}, RETENTION_UPLIFT={RETENTION_UPLIFT}")
    print("EV@K (expected):", ev_at_k)
    print("Retention campaign simulation (profit):", sim_at_k)

    # --------- Drift reference snapshot (train distributions) ---------
    # Build feature-frame after align/clean/feat (categoricals preserved).
    try:
        # Для фичей нужен именно базовый pipeline (preprocess/feat),
        # а не калибратор-обёртка.
        X_train_feat = build_feature_frame(base_pipe, X_train)
        X_test_feat = build_feature_frame(base_pipe, X_test)
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
            if edges.size >= 3:
                # ref bin probs (for PSI/KS in API)
                idx = np.digitize(proba_ref, edges[1:-1], right=False)
                counts = np.bincount(idx, minlength=edges.size - 1).astype(float)
                probs = counts / counts.sum() if counts.sum() > 0 else np.zeros(edges.size - 1, dtype=float)
                pred_ref = {
                    "mean": float(np.mean(proba_ref)),
                    "std": float(np.std(proba_ref, ddof=0)),
                    "edges": [float(x) for x in edges.tolist()],
                    "ref_bin_probs": [float(x) for x in probs.tolist()],
                }
            else:
                pred_ref = {
                    "mean": float(np.mean(proba_ref)),
                    "std": float(np.std(proba_ref, ddof=0)),
                    "edges": None,
                    "ref_bin_probs": None,
                }
        else:
            pred_ref = {"mean": None, "std": None, "edges": None, "ref_bin_probs": None}
    except Exception:
        pred_ref = {"error": "failed_to_build_prediction_reference"}

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
        # legacy config threshold (не используется для решения, но полезно хранить)
        "threshold_config": THRESHOLD,
        # выбранный порог по max EV
        "selected_threshold": selected_threshold,
        # expected uplift (baseline=0, uplift==EV)
        "expected_uplift": expected_uplift,
        # бизнес-параметры EV/симуляции (для воспроизводимости)
        "ltv_saved": LTV_SAVED,
        "cost_action": COST_ACTION,
        "retention_uplift": RETENTION_UPLIFT,
        # бизнес-метрики на holdout
        "ev_at_k": ev_at_k,
        "retention_simulation_at_k": sim_at_k,
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
