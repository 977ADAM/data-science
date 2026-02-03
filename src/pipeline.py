# src/pipeline.py

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier

from src.preprocessing import Cleaner


# ---------- Custom transformers ----------

def _make_ohe() -> OneHotEncoder:
    """
    Совместимость sklearn:
    - старые версии: OneHotEncoder(sparse=...)
    - новые версии: OneHotEncoder(sparse_output=...)
    Делаем плотный вывод, чтобы избежать сюрпризов с поддержкой sparse у моделей/сериализации.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)

class AlignColumns(BaseEstimator, TransformerMixin):
    """
    Гарантирует стабильный набор входных колонок между train и inference.
    - На fit запоминает ожидаемые колонки.
    - На transform добавляет отсутствующие как NaN и отбрасывает лишние.
    Это критично, когда API присылает не все поля, которые были в train.
    """
    def __init__(self, expected_columns: list[str]):
        self.expected_columns = list(expected_columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # добавить отсутствующие
        for c in self.expected_columns:
            if c not in df.columns:
                df[c] = np.nan
        # оставить только ожидаемые и в правильном порядке
        return df[self.expected_columns]

class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Бизнес-фичи для churn/маркетинга.
    Важно: делаем безопасно, чтобы API не падал, если не пришли какие-то поля.
    """
    def __init__(self):
        # threshold for high_charges; learned on train in fit()
        self.monthly_median_ = None


    def fit(self, X, y=None):
        df = X.copy()
        if "MonthlyCharges" in df.columns:
            s = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
            self.monthly_median_ = float(s.median()) if s.notna().any() else 0.0
        else:
            self.monthly_median_ = 0.0
        return self

    def transform(self, X):
        df = X.copy()

        # df.get(..., scalar) может вернуть скаляр, у которого нет .fillna().
        # Делаем гарантированно Series нужной длины (важно для robustness на инференсе).
        if "tenure" in df.columns:
            tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        else:
            tenure = pd.Series(0, index=df.index, dtype="float64")

        if "TotalCharges" in df.columns:
            total = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
        else:
            total = pd.Series(0.0, index=df.index, dtype="float64")

        if "MonthlyCharges" in df.columns:
            monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
        else:
            monthly = pd.Series(0.0, index=df.index, dtype="float64")

        # Денежные/поведенческие
        df["avg_monthly_bill"] = total / (tenure + 1)

        if "MonthlyCharges" in df.columns:
            # monthly может быть серией; сравнение векторное
            threshold = 0.0 if self.monthly_median_ is None else float(self.monthly_median_)
            df["high_charges"] = (
                monthly > threshold
            ).astype(int)
        else:
            df["high_charges"] = 0

        # Время жизни
        df["is_new_client"] = (tenure < 12).astype(int)
        df["is_long_term"] = (tenure > 24).astype(int)

        # Bucket tenure (для маркетинга/сегментов)
        # если tenure нет — будет 0
        df["tenure_bucket"] = (
            pd.cut(
                tenure,
                bins=[-1, 12, 24, 48, 72, np.inf],
                labels=False
            )
            .fillna(0)
            .astype(int)
        )

        # Количество подключенных сервисов
        services = [
            "PhoneService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        num_services = 0
        for s in services:
            if s in df.columns:
                # Более устойчиво к "yes"/" Yes " и пр. вариациям на инференсе
                num_services += (
                    df[s]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("yes")
                ).astype(int)
        df["num_services"] = num_services

        # Контракт/оплата
        if "Contract" in df.columns:
            contract = df["Contract"].astype(str).str.strip()
            df["is_month_to_month"] = contract.eq("Month-to-month").astype(int)
        else:
            df["is_month_to_month"] = 0

        if "PaymentMethod" in df.columns:
            pm = df["PaymentMethod"].fillna("").astype(str)
            df["is_auto_pay"] = pm.str.contains("automatic", case=False, na=False).astype(int)
        else:
            df["is_auto_pay"] = 0

        # Доходность (условная)
        df["revenue_per_tenure"] = total / (tenure + 1)

        return df

class SklearnCatBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper around catboost.CatBoostClassifier.

    Why: sklearn>=1.6 relies on sklearn BaseEstimator API (params/tags) in pipelines.
    Native CatBoostClassifier may not implement sklearn tags API, which causes
    AttributeError: object has no attribute '_sklearn_tags_' during predict_proba().
    """
    def __init__(self, **catboost_params):
        self.catboost_params = dict(catboost_params)

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(**self.catboost_params)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    # sklearn uses this to decide if estimator is fitted in some contexts
    def __sklearn_is_fitted__(self):
        return hasattr(self, "model_")

    def _more_tags(self):
        # минимальные теги для совместимости со sklearn-пайплайнами/проверками
        return {"requires_y": True}

    # expose common attributes if somebody uses them downstream
    @property
    def feature_importances_(self):
        return getattr(self.model_, "feature_importances_", None)

    def get_params(self, deep=True):
        """
        Делает estimator sklearn-compatible:
        - параметры доступны "плоско" (model__iterations, model__depth, ...)
        - clone()/GridSearchCV работают ожидаемо
        """
        return dict(self.catboost_params)

    def set_params(self, **params):
        """
        Поддержка обоих стилей:
        - set_params(iterations=..., depth=...)
        - set_params(catboost_params={...})  (на всякий случай)
        """
        if "catboost_params" in params:
            cb = params.pop("catboost_params")
            if cb is not None:
                self.catboost_params = dict(cb)
        if params:
            self.catboost_params.update(params)
        return self

def make_pipeline(X_sample: pd.DataFrame) -> Pipeline:
    """
    Создаём pipeline на основе примера данных.
    """
    # service columns / id columns можно удалить сразу
    drop_cols = [c for c in ["customerID", "Churn"] if c in X_sample.columns]
    X_sample = X_sample.drop(columns=drop_cols)
    expected_raw_cols = list(X_sample.columns)

    # определяем типы колонок ПОСЛЕ feature engineering
    # поэтому сначала прогоняем через cleaner+features на сэмпле
    # (fit трансформеров нужен, т.к. в FeatureBuilder есть обучаемые параметры)
    cleaner = Cleaner()
    fb = FeatureBuilder()
    Xc = cleaner.fit_transform(X_sample)
    tmp = fb.fit(Xc).transform(Xc)

    numeric_cols = [c for c in tmp.columns if pd.api.types.is_numeric_dtype(tmp[c])]
    categorical_cols = [c for c in tmp.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", _make_ohe(), categorical_cols),
        ],
        remainder="drop",
    )

    model = SklearnCatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=0,
        random_seed=42,
        # Для воспроизводимости (многопоток часто даёт недетерминизм).
        # В проде для скорости можно обучать отдельно и фиксировать артефакт,
        # но для гарантированного воспроизводимого обучения оставляем 1.
        thread_count=1,
        allow_writing_files=False,
    )

    pipe = Pipeline(steps=[
        ("align", AlignColumns(expected_raw_cols)),
        ("clean", Cleaner()),
        ("feat", FeatureBuilder()),
        ("prep", preprocessor),
        ("model", model),
    ])

    return pipe
