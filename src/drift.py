from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def _safe_series_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def _ks_stat_and_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """
    Two-sample Kolmogorov–Smirnov test (asymptotic p-value approximation).
    We avoid scipy dependency; p-value is a standard asymptotic approximation.
    """
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
    # empirical CDFs
    cdf1 = np.searchsorted(x_sort, data_all, side="right") / n1
    cdf2 = np.searchsorted(y_sort, data_all, side="right") / n2
    d = float(np.max(np.abs(cdf1 - cdf2)))

    # Asymptotic p-value (Kolmogorov distribution)
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d
    # Q_KS(lam) ≈ 2 * Σ (-1)^(j-1) exp(-2 j^2 lam^2)
    s = 0.0
    for j in range(1, 101):
        term = (-1) ** (j - 1) * math.exp(-2.0 * (j * lam) ** 2)
        s += term
        if abs(term) < 1e-10:
            break
    p = max(0.0, min(1.0, 2.0 * s))
    return d, p


def _psi_from_probs(ref_p: np.ndarray, cur_p: np.ndarray, eps: float = 1e-6) -> float | None:
    ref_p = np.asarray(ref_p, dtype=float)
    cur_p = np.asarray(cur_p, dtype=float)
    if ref_p.size == 0 or cur_p.size == 0 or ref_p.size != cur_p.size:
        return None
    ref_p = np.clip(ref_p, eps, 1.0)
    cur_p = np.clip(cur_p, eps, 1.0)
    ref_p = ref_p / ref_p.sum()
    cur_p = cur_p / cur_p.sum()
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _bin_probs(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # bins are [e0,e1),[e1,e2),...,[e_{k-1},e_k] where last is inclusive
    if edges.size < 2:
        return np.array([], dtype=float)
    # Make sure edges are strictly increasing; otherwise bincount is unstable
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([], dtype=float)
    # np.digitize: returns 1..len(edges)-1
    idx = np.digitize(values, edges[1:-1], right=False)
    counts = np.bincount(idx, minlength=edges.size - 1).astype(float)
    total = counts.sum()
    if total <= 0:
        return np.zeros(edges.size - 1, dtype=float)
    return counts / total


def build_feature_frame(pipe: Any, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Try to apply the same (align->clean->feat) transformations as in training.
    If pipe structure is unknown (e.g. DummyPipe in tests), return df_raw as-is.
    """
    df = df_raw.copy()
    try:
        # CalibratedClassifierCV -> base_estimator is the Pipeline
        base = getattr(pipe, "base_estimator", None) or getattr(pipe, "estimator", None) or pipe
        named_steps = getattr(base, "named_steps", None)
        if not named_steps:
            return df

        for step_name in ("align", "clean", "feat"):
            step = named_steps.get(step_name)
            if step is None:
                return df
            df = step.transform(df)
        if not isinstance(df, pd.DataFrame):
            return df_raw.copy()
        return df
    except Exception:
        return df_raw.copy()


def build_reference_profile(
    df_features: pd.DataFrame,
    *,
    numeric_bins: int = 10,
    max_categories: int = 50,
) -> dict[str, Any]:
    """
    Reference distribution snapshot.
    - numeric: mean/std + quantile edges + ref bin probabilities (PSI)
    - categorical: normalized frequency map (top-K + __OTHER__)
    """
    out: dict[str, Any] = {
        "numeric": {},
        "categorical": {},
        "meta": {
            "rows": int(len(df_features)),
            "numeric_bins": int(numeric_bins),
            "max_categories": int(max_categories),
        },
    }

    for col in df_features.columns:
        s = df_features[col]
        if pd.api.types.is_numeric_dtype(s):
            sn = _safe_series_numeric(s)
            if sn.empty:
                out["numeric"][col] = {"mean": None, "std": None, "edges": None, "ref_bin_probs": None}
                continue

            vals = sn.to_numpy(dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=0))
            qs = np.linspace(0.0, 1.0, numeric_bins + 1)
            edges = np.quantile(vals, qs)
            edges = np.unique(edges)
            if edges.size < 3:
                # Too few unique edges -> PSI bins are meaningless
                out["numeric"][col] = {"mean": mean, "std": std, "edges": None, "ref_bin_probs": None}
                continue
            ref_probs = _bin_probs(vals, edges)
            out["numeric"][col] = {
                "mean": mean,
                "std": std,
                "edges": [float(x) for x in edges.tolist()],
                "ref_bin_probs": [float(x) for x in ref_probs.tolist()],
            }
        else:
            # categorical: keep strings stable, include missing bucket
            sc = s.astype("object")
            sc = sc.where(sc.notna(), "__MISSING__").astype(str)
            vc = sc.value_counts(dropna=False)
            total = float(vc.sum()) if float(vc.sum()) > 0 else 1.0
            probs = (vc / total).to_dict()
            # keep top-K
            items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            top = items[:max_categories]
            other_prob = float(sum(p for _, p in items[max_categories:])) if len(items) > max_categories else 0.0
            freq: dict[str, float] = {str(k): float(v) for k, v in top}
            if other_prob > 0:
                freq["__OTHER__"] = other_prob
            out["categorical"][col] = {"freq": freq}

    return out


def compare_to_reference(
    df_features: pd.DataFrame,
    reference: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute drift metrics:
    - numeric: PSI + KS-statistic/pvalue + mean/std deltas
    - categorical: L1 distance + categorical PSI (freq vs freq)
    """
    results: dict[str, Any] = {"numeric": {}, "categorical": {}, "meta": {"rows": int(len(df_features))}}

    # ---------- numeric ----------
    ref_num: dict[str, Any] = (reference or {}).get("numeric") or {}
    for col, ref in ref_num.items():
        if col not in df_features.columns:
            results["numeric"][col] = {"present": False}
            continue
        s = df_features[col]
        sn = _safe_series_numeric(s)
        if sn.empty:
            results["numeric"][col] = {"present": True, "psi": None, "ks_stat": None, "ks_pvalue": None}
            continue

        cur_vals = sn.to_numpy(dtype=float)
        ref_mean = _to_float(ref.get("mean"))
        ref_std = _to_float(ref.get("std"))
        cur_mean = float(np.mean(cur_vals))
        cur_std = float(np.std(cur_vals, ddof=0))

        edges = ref.get("edges")
        ref_probs = ref.get("ref_bin_probs")
        psi = None
        if edges and ref_probs:
            e = np.asarray(edges, dtype=float)
            rp = np.asarray(ref_probs, dtype=float)
            cp = _bin_probs(cur_vals, e)
            if cp.size == rp.size and cp.size > 0:
                psi = _psi_from_probs(rp, cp)

        # KS: compare current vs synthetic from reference? No -> use reference bins not enough.
        # We expect to have reference raw values? Not stored. So KS is computed vs reference bin-approx is not valid.
        # Instead, store reference_values_sample? Not required; but requirement says KS test.
        # We approximate KS by sampling from bins using midpoints & ref probs.
        ks_stat, ks_p = None, None
        try:
            if edges and ref_probs:
                e = np.asarray(edges, dtype=float)
                rp = np.asarray(ref_probs, dtype=float)
                # sample synthetic reference (size = min(5000, cur_n*2))
                n_syn = int(min(5000, max(200, cur_vals.size * 2)))
                # build midpoints
                mids = (e[:-1] + e[1:]) / 2.0
                if mids.size == rp.size and mids.size > 0 and rp.sum() > 0:
                    syn = np.random.default_rng(42).choice(mids, size=n_syn, p=rp / rp.sum())
                    ks_stat, ks_p = _ks_stat_and_pvalue(syn, cur_vals)
        except Exception:
            ks_stat, ks_p = None, None

        results["numeric"][col] = {
            "present": True,
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "ref_mean": ref_mean,
            "cur_mean": cur_mean,
            "mean_delta": None if ref_mean is None else float(cur_mean - ref_mean),
            "ref_std": ref_std,
            "cur_std": cur_std,
            "std_delta": None if ref_std is None else float(cur_std - ref_std),
        }

    # ---------- categorical ----------
    ref_cat: dict[str, Any] = (reference or {}).get("categorical") or {}
    for col, ref in ref_cat.items():
        if col not in df_features.columns:
            results["categorical"][col] = {"present": False}
            continue
        s = df_features[col].astype("object")
        s = s.where(s.notna(), "__MISSING__").astype(str)
        vc = s.value_counts(dropna=False)
        total = float(vc.sum()) if float(vc.sum()) > 0 else 1.0
        cur_freq = (vc / total).to_dict()

        ref_freq: dict[str, float] = ((ref or {}).get("freq") or {}).copy()
        if not ref_freq:
            results["categorical"][col] = {"present": True, "l1": None, "psi": None}
            continue

        # map current categories into reference support (+ unseen bucket)
        cur_mapped: dict[str, float] = {}
        unseen = 0.0
        for k, p in cur_freq.items():
            kk = str(k)
            if kk in ref_freq:
                cur_mapped[kk] = cur_mapped.get(kk, 0.0) + float(p)
            else:
                unseen += float(p)
        if unseen > 0:
            if "__OTHER__" in ref_freq:
                cur_mapped["__OTHER__"] = cur_mapped.get("__OTHER__", 0.0) + unseen
            else:
                cur_mapped["__UNSEEN__"] = unseen

        # align vectors
        keys = sorted(set(ref_freq.keys()) | set(cur_mapped.keys()))
        ref_p = np.array([float(ref_freq.get(k, 0.0)) for k in keys], dtype=float)
        cur_p = np.array([float(cur_mapped.get(k, 0.0)) for k in keys], dtype=float)
        # normalize
        if ref_p.sum() > 0:
            ref_p = ref_p / ref_p.sum()
        if cur_p.sum() > 0:
            cur_p = cur_p / cur_p.sum()

        l1 = float(0.5 * np.sum(np.abs(cur_p - ref_p)))
        psi = _psi_from_probs(ref_p, cur_p)

        results["categorical"][col] = {
            "present": True,
            "l1": l1,
            "psi": psi,
            "unseen_mass": float(unseen),
        }

    return results
