# src/uplift.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _as_1d_int(a: Any) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x.astype(int)

def _as_1d_float(a: Any) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim != 1:
        x = x.reshape(-1)
    return x.astype(float)

def _slice_rows(X, mask: np.ndarray):
    """
    Row-slice helper that works for both pandas.DataFrame and numpy arrays.
    Pandas gotcha: X[mask] slices columns, not rows (unless mask is an aligned Series).
    """
    m = np.asarray(mask, dtype=bool).reshape(-1)
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            # mask is positional; use iloc
            return X.iloc[m]
    except Exception:
        pass
    return np.asarray(X)[m]

def _predict_proba_pos(estimator: Any, X) -> np.ndarray:
    """
    Robustly get positive-class probabilities for binary outcome models.
    Supports:
      - sklearn-like predict_proba -> [:,1]
      - fallback: predict (already in [0,1] or 0/1)
    """
    if hasattr(estimator, "predict_proba"):
        p = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
    else:
        p = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    return p


def _predict_regression(estimator: Any, X) -> np.ndarray:
    """
    Predict continuous target (tau, pseudo-outcome, etc.)
    Supports predict(); if predict_proba exists, uses pos proba.
    """
    # If estimator exposes probabilities (e.g., classifier used as effect model),
    # prefer positive-class probability as a continuous signal.
    if hasattr(estimator, "predict_proba"):
        return _predict_proba_pos(estimator, X).reshape(-1)
    return np.asarray(estimator.predict(X), dtype=float).reshape(-1)


def _augment_with_treatment(X, t: np.ndarray, *, treatment_col: str = "treatment"):
    """
    Add treatment indicator as a feature.
    - If X is a pandas.DataFrame, add a new column.
    - Else, append as last numeric column to numpy array.
    """
    try:
        import pandas as pd  # optional dependency in this module

        if isinstance(X, pd.DataFrame):
            df = X.copy()
            df[treatment_col] = np.asarray(t, dtype=int)
            return df
    except Exception:
        pass
    x = np.asarray(X)
    tt = np.asarray(t, dtype=int).reshape(-1, 1)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.concatenate([x, tt], axis=1)

@dataclass
class TLearnerArtifacts:
    """Lightweight container for the two models."""

    model_treated: Any
    model_control: Any


class TLearner:
    """
    Two-model uplift (T-learner):
      uplift(x) = P(Y=1|X=x, T=1) - P(Y=1|X=x, T=0)

    Notes:
    - Assumes binary treatment indicator (0/1) and binary outcome.
    - Uses predict_proba[:,1] as positive class probability.
    - Keeps sklearn-like API so it can be pickled/joblib-dumped.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        *,
        clip: bool = True,
    ):
        self.estimator_factory = estimator_factory
        self.clip = bool(clip)
        self.artifacts_: TLearnerArtifacts | None = None

    def fit(self, X, treatment, y):
        t = _as_1d_int(treatment)
        y = _as_1d_int(y)
        if t.size != y.size:
            raise ValueError("treatment and y must have the same length")

        mask_t = t == 1
        mask_c = t == 0
        if not np.any(mask_t) or not np.any(mask_c):
            raise ValueError("Both treated (t=1) and control (t=0) groups must be present")

        mt = self.estimator_factory()
        mc = self.estimator_factory()

        Xt = _slice_rows(X, mask_t)
        Xc = _slice_rows(X, mask_c)
        mt.fit(Xt, y[mask_t])
        mc.fit(Xc, y[mask_c])

        self.artifacts_ = TLearnerArtifacts(model_treated=mt, model_control=mc)
        return self

    def _require_fitted(self) -> TLearnerArtifacts:
        art = getattr(self, "artifacts_", None)
        if art is None:
            raise AttributeError("TLearner is not fitted. Call fit() first.")
        return art

    def predict_proba_treated(self, X) -> np.ndarray:
        art = self._require_fitted()
        p = np.asarray(art.model_treated.predict_proba(X)[:, 1], dtype=float)
        if self.clip:
            p = np.clip(p, 0.0, 1.0)
        return p

    def predict_proba_control(self, X) -> np.ndarray:
        art = self._require_fitted()
        p = np.asarray(art.model_control.predict_proba(X)[:, 1], dtype=float)
        if self.clip:
            p = np.clip(p, 0.0, 1.0)
        return p

    def predict_uplift(self, X) -> np.ndarray:
        pt = self.predict_proba_treated(X)
        pc = self.predict_proba_control(X)
        u = pt - pc
        if self.clip:
            # uplift can be negative; clip only to [-1,1]
            u = np.clip(u, -1.0, 1.0)
        return u

    # sklearn-like convenience
    def get_params(self, deep: bool = True):
        return {"estimator_factory": self.estimator_factory, "clip": self.clip}

    def set_params(self, **params):
        if "estimator_factory" in params:
            self.estimator_factory = params.pop("estimator_factory")
        if "clip" in params:
            self.clip = bool(params.pop("clip"))
        return self
    


@dataclass
class SLearnerArtifacts:
    """Single-model uplift artifacts."""

    model: Any
    treatment_col: str


class SLearner:
    """
    Single-model uplift (S-learner) baseline:
      - Train one outcome model on features [X, T]
      - Score by toggling T=1 vs T=0:
          uplift(x) = P(Y=1|X=x,T=1) - P(Y=1|X=x,T=0)
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        *,
        treatment_col: str = "treatment",
        clip: bool = True,
    ):
        self.estimator_factory = estimator_factory
        self.treatment_col = str(treatment_col)
        self.clip = bool(clip)
        self.artifacts_: SLearnerArtifacts | None = None

    def fit(self, X, treatment, y):
        t = _as_1d_int(treatment)
        y = _as_1d_int(y)
        if t.size != y.size:
            raise ValueError("treatment and y must have the same length")
        if not set(np.unique(t)).issubset({0, 1}):
            raise ValueError("treatment must be binary 0/1")

        Xm = _augment_with_treatment(X, t, treatment_col=self.treatment_col)
        m = self.estimator_factory()
        m.fit(Xm, y)
        self.artifacts_ = SLearnerArtifacts(model=m, treatment_col=self.treatment_col)
        return self

    def _require_fitted(self) -> SLearnerArtifacts:
        art = getattr(self, "artifacts_", None)
        if art is None:
            raise AttributeError("SLearner is not fitted. Call fit() first.")
        return art

    def predict_proba_treated(self, X) -> np.ndarray:
        art = self._require_fitted()
        t1 = np.ones(len(X), dtype=int)
        Xm = _augment_with_treatment(X, t1, treatment_col=art.treatment_col)
        p = _predict_proba_pos(art.model, Xm)
        if self.clip:
            p = np.clip(p, 0.0, 1.0)
        return p

    def predict_proba_control(self, X) -> np.ndarray:
        art = self._require_fitted()
        t0 = np.zeros(len(X), dtype=int)
        Xm = _augment_with_treatment(X, t0, treatment_col=art.treatment_col)
        p = _predict_proba_pos(art.model, Xm)
        if self.clip:
            p = np.clip(p, 0.0, 1.0)
        return p

    def predict_uplift(self, X) -> np.ndarray:
        u = self.predict_proba_treated(X) - self.predict_proba_control(X)
        if self.clip:
            u = np.clip(u, -1.0, 1.0)
        return u

    def get_params(self, deep: bool = True):
        return {
            "estimator_factory": self.estimator_factory,
            "treatment_col": self.treatment_col,
            "clip": self.clip,
        }

    def set_params(self, **params):
        if "estimator_factory" in params:
            self.estimator_factory = params.pop("estimator_factory")
        if "treatment_col" in params:
            self.treatment_col = str(params.pop("treatment_col"))
        if "clip" in params:
            self.clip = bool(params.pop("clip"))
        return self


@dataclass
class XLeanerArtifacts:
    model_mu_treated: Any
    model_mu_control: Any
    model_tau_treated: Any
    model_tau_control: Any
    model_propensity: Any


class XLearner:
    """
    X-learner (Künzel et al.):
      1) Fit outcome models mu1(x), mu0(x)
      2) Impute effects:
           D1 = y - mu0(x) for treated
           D0 = mu1(x) - y for control
      3) Fit effect models tau1(x) on treated with target D1
                     tau0(x) on control with target D0
      4) Fit propensity e(x)=P(T=1|X)
      5) Combine:
           tau(x) = e(x)*tau0(x) + (1-e(x))*tau1(x)
    """

    def __init__(
        self,
        *,
        outcome_model_factory: Callable[[], Any],
        effect_model_factory: Callable[[], Any],
        propensity_model_factory: Callable[[], Any],
        clip: bool = True,
        propensity_eps: float = 1e-3,
    ):
        self.outcome_model_factory = outcome_model_factory
        self.effect_model_factory = effect_model_factory
        self.propensity_model_factory = propensity_model_factory
        self.clip = bool(clip)
        self.propensity_eps = float(propensity_eps)
        self.artifacts_: XLeanerArtifacts | None = None

    def fit(self, X, treatment, y):
        t = _as_1d_int(treatment)
        y = _as_1d_int(y)
        if t.size != y.size:
            raise ValueError("treatment and y must have the same length")
        if not set(np.unique(t)).issubset({0, 1}):
            raise ValueError("treatment must be binary 0/1")

        mask_t = t == 1
        mask_c = t == 0
        if not np.any(mask_t) or not np.any(mask_c):
            raise ValueError("Both treated (t=1) and control (t=0) groups must be present")

        # 1) outcome models
        mu1 = self.outcome_model_factory()
        mu0 = self.outcome_model_factory()
        Xt = _slice_rows(X, mask_t)
        Xc = _slice_rows(X, mask_c)
        mu1.fit(Xt, y[mask_t])
        mu0.fit(Xc, y[mask_c])

        # 2) imputed effects
        mu0_all = _predict_proba_pos(mu0, X)
        mu1_all = _predict_proba_pos(mu1, X)
        d1 = (y[mask_t].astype(float) - mu0_all[mask_t]).astype(float)
        d0 = (mu1_all[mask_c] - y[mask_c].astype(float)).astype(float)

        # 3) effect models
        tau1 = self.effect_model_factory()
        tau0 = self.effect_model_factory()
        tau1.fit(Xt, d1)
        tau0.fit(Xc, d0)

        # 4) propensity
        e = self.propensity_model_factory()
        e.fit(X, t)

        self.artifacts_ = XLeanerArtifacts(
            model_mu_treated=mu1,
            model_mu_control=mu0,
            model_tau_treated=tau1,
            model_tau_control=tau0,
            model_propensity=e,
        )
        return self

    def _require_fitted(self) -> XLeanerArtifacts:
        art = getattr(self, "artifacts_", None)
        if art is None:
            raise AttributeError("XLearner is not fitted. Call fit() first.")
        return art

    def predict_uplift(self, X) -> np.ndarray:
        art = self._require_fitted()
        e = _predict_proba_pos(art.model_propensity, X)
        eps = self.propensity_eps
        e = np.clip(e, eps, 1.0 - eps)

        tau1 = _predict_regression(art.model_tau_treated, X)
        tau0 = _predict_regression(art.model_tau_control, X)
        tau = e * tau0 + (1.0 - e) * tau1

        if self.clip:
            tau = np.clip(tau, -1.0, 1.0)
        return tau

    def predict_proba_treated(self, X) -> np.ndarray:
        art = self._require_fitted()
        mu0 = _predict_proba_pos(art.model_mu_control, X)
        tau = self.predict_uplift(X)
        pt = mu0 + tau
        if self.clip:
            pt = np.clip(pt, 0.0, 1.0)
        return pt

    def predict_proba_control(self, X) -> np.ndarray:
        art = self._require_fitted()
        mu0 = _predict_proba_pos(art.model_mu_control, X)
        if self.clip:
            mu0 = np.clip(mu0, 0.0, 1.0)
        return mu0

    def get_params(self, deep: bool = True):
        return {
            "outcome_model_factory": self.outcome_model_factory,
            "effect_model_factory": self.effect_model_factory,
            "propensity_model_factory": self.propensity_model_factory,
            "clip": self.clip,
            "propensity_eps": self.propensity_eps,
        }

    def set_params(self, **params):
        if "outcome_model_factory" in params:
            self.outcome_model_factory = params.pop("outcome_model_factory")
        if "effect_model_factory" in params:
            self.effect_model_factory = params.pop("effect_model_factory")
        if "propensity_model_factory" in params:
            self.propensity_model_factory = params.pop("propensity_model_factory")
        if "clip" in params:
            self.clip = bool(params.pop("clip"))
        if "propensity_eps" in params:
            self.propensity_eps = float(params.pop("propensity_eps"))
        return self


@dataclass
class DRLearnerArtifacts:
    model_mu_treated: Any
    model_mu_control: Any
    model_propensity: Any
    model_tau: Any


class DRLearner:
    """
    Doubly Robust (DR) learner:
      - Fit outcome models mu1(x), mu0(x)
      - Fit propensity e(x)=P(T=1|X)
      - Build pseudo-outcome (AIPW / EIF):
          phi = (t*(y-mu1)/e) - ((1-t)*(y-mu0)/(1-e)) + (mu1 - mu0)
      - Regress phi on X to estimate tau(x)
    """

    def __init__(
        self,
        *,
        outcome_model_factory: Callable[[], Any],
        propensity_model_factory: Callable[[], Any],
        effect_model_factory: Callable[[], Any],
        clip: bool = True,
        propensity_eps: float = 1e-3,
    ):
        self.outcome_model_factory = outcome_model_factory
        self.propensity_model_factory = propensity_model_factory
        self.effect_model_factory = effect_model_factory
        self.clip = bool(clip)
        self.propensity_eps = float(propensity_eps)
        self.artifacts_: DRLearnerArtifacts | None = None

    def fit(self, X, treatment, y):
        t = _as_1d_int(treatment)
        y = _as_1d_int(y)
        if t.size != y.size:
            raise ValueError("treatment and y must have the same length")
        if not set(np.unique(t)).issubset({0, 1}):
            raise ValueError("treatment must be binary 0/1")

        mask_t = t == 1
        mask_c = t == 0
        if not np.any(mask_t) or not np.any(mask_c):
            raise ValueError("Both treated (t=1) and control (t=0) groups must be present")

        mu1 = self.outcome_model_factory()
        mu0 = self.outcome_model_factory()
        Xt = _slice_rows(X, mask_t)
        Xc = _slice_rows(X, mask_c)
        mu1.fit(Xt, y[mask_t])
        mu0.fit(Xc, y[mask_c])

        e = self.propensity_model_factory()
        e.fit(X, t)

        p1 = _predict_proba_pos(mu1, X)
        p0 = _predict_proba_pos(mu0, X)
        prop = _predict_proba_pos(e, X)
        eps = self.propensity_eps
        prop = np.clip(prop, eps, 1.0 - eps)

        y_f = y.astype(float)
        t_f = t.astype(float)
        # DR pseudo outcome (EIF for ATE, pointwise)
        phi = (t_f * (y_f - p1) / prop) - ((1.0 - t_f) * (y_f - p0) / (1.0 - prop)) + (p1 - p0)

        tau = self.effect_model_factory()
        tau.fit(X, phi.astype(float))

        self.artifacts_ = DRLearnerArtifacts(
            model_mu_treated=mu1,
            model_mu_control=mu0,
            model_propensity=e,
            model_tau=tau,
        )
        return self

    def _require_fitted(self) -> DRLearnerArtifacts:
        art = getattr(self, "artifacts_", None)
        if art is None:
            raise AttributeError("DRLearner is not fitted. Call fit() first.")
        return art

    def predict_uplift(self, X) -> np.ndarray:
        art = self._require_fitted()
        tau = _predict_regression(art.model_tau, X)
        if self.clip:
            tau = np.clip(tau, -1.0, 1.0)
        return tau

    def predict_proba_treated(self, X) -> np.ndarray:
        art = self._require_fitted()
        p0 = _predict_proba_pos(art.model_mu_control, X)
        pt = p0 + self.predict_uplift(X)
        if self.clip:
            pt = np.clip(pt, 0.0, 1.0)
        return pt

    def predict_proba_control(self, X) -> np.ndarray:
        art = self._require_fitted()
        p0 = _predict_proba_pos(art.model_mu_control, X)
        if self.clip:
            p0 = np.clip(p0, 0.0, 1.0)
        return p0

    def get_params(self, deep: bool = True):
        return {
            "outcome_model_factory": self.outcome_model_factory,
            "propensity_model_factory": self.propensity_model_factory,
            "effect_model_factory": self.effect_model_factory,
            "clip": self.clip,
            "propensity_eps": self.propensity_eps,
        }

    def set_params(self, **params):
        if "outcome_model_factory" in params:
            self.outcome_model_factory = params.pop("outcome_model_factory")
        if "propensity_model_factory" in params:
            self.propensity_model_factory = params.pop("propensity_model_factory")
        if "effect_model_factory" in params:
            self.effect_model_factory = params.pop("effect_model_factory")
        if "clip" in params:
            self.clip = bool(params.pop("clip"))
        if "propensity_eps" in params:
            self.propensity_eps = float(params.pop("propensity_eps"))
        return self
