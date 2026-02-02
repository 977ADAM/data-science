from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class HoldoutCalibratedClassifier:
    """
    Calibrate probabilities on a separate holdout (no CV).

    Why:
    - Avoid CalibratedClassifierCV classes_/CV quirks
    - Keep artifact stable and sklearn-like
    """

    def __init__(self, estimator, method: str = "isotonic"):
        self.estimator = estimator
        self.method = method
        self.calibrator_ = None
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X_cal, y_cal):
        p = self.estimator.predict_proba(X_cal)[:, 1]
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        y_cal = np.asarray(y_cal, dtype=int)

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y_cal)
            self.calibrator_ = ("isotonic", iso)
        elif self.method == "sigmoid":
            # Platt scaling: logit(prob) -> logistic regression
            logit = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(logit, y_cal)
            self.calibrator_ = ("sigmoid", lr)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        return self

    def _calibrate_pos(self, p):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        name, cal = self.calibrator_
        if name == "isotonic":
            p_cal = cal.transform(p)
        else:
            logit = np.log(p / (1.0 - p)).reshape(-1, 1)
            p_cal = cal.predict_proba(logit)[:, 1]
        return np.clip(np.asarray(p_cal, dtype=float), 0.0, 1.0)

    def predict_proba(self, X):
        p = self.estimator.predict_proba(X)[:, 1]
        p_cal = self._calibrate_pos(p)
        return np.column_stack([1.0 - p_cal, p_cal])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)