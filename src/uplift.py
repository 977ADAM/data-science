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

        mt.fit(X[mask_t], y[mask_t])
        mc.fit(X[mask_c], y[mask_c])

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
