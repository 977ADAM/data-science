# src/preprocessing.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # таргет может отсутствовать на инференсе
    if "Churn" in df.columns:
        s = df["Churn"]
        # приводим к устойчивому виду: strip + lower
        s = s.astype(str).str.strip().str.lower()
        df["Churn"] = s.isin(["yes", "y", "1", "true", "t"]).astype(int)
    return df


class Cleaner(BaseEstimator, TransformerMixin):
    """
    Минимальная очистка.
    - TotalCharges -> numeric
    - заполняем пропуски (простая стратегия)
    """
    def __init__(self):
        self.num_fill_ = {}
        self.cat_fill_ = {}

    def fit(self, X, y=None):

        df = X.copy()
        self.numeric_cols_ = []
        self.categorical_cols_ = []
        # фиксируем статистики на трейне, чтобы импьют не "плавал" на инференсе
        self.num_fill_.clear()
        self.cat_fill_.clear()
        def _is_mostly_numeric(series: pd.Series, min_ratio: float = 0.9) -> bool:
            """
            True если колонка "в основном" числовая, даже если dtype=object (числа строками).
            min_ratio — доля успешно распарсенных значений среди non-null.
            """
            # быстрый путь: уже numeric dtype
            if pd.api.types.is_numeric_dtype(series):
                return True
            # пробуем парсинг
            s_num = pd.to_numeric(series, errors="coerce")
            non_null = series.notna().sum()
            if non_null == 0:
                return False
            ratio = s_num.notna().sum() / non_null
            return ratio >= min_ratio

        for col in df.columns:
            s = df[col]

            # TotalCharges часто приходит как строка с пробелами
            if col == "TotalCharges":
                s = pd.to_numeric(s, errors="coerce")

            if _is_mostly_numeric(s):
                self.numeric_cols_.append(col)
                s_num = pd.to_numeric(s, errors="coerce")
                self.num_fill_[col] = float(s_num.median()) if s_num.notna().any() else 0.0
            else:
                self.categorical_cols_.append(col)
                # простая стратегия для категорий: mode, иначе "unknown"
                # важно: нормализуем строки УЖЕ на fit, иначе mode может стать "" / "nan" и т.п.
                if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                    s = s.astype(str).str.strip()
                    # аккуратно считаем "псевдо-NaN" без изменения регистра реальных категорий
                    s_lower = s.str.lower()
                    mask_na = s_lower.isin(["", "nan", "none", "null"])
                    s = s.mask(mask_na, np.nan)               
                try:
                    mode = s.dropna().mode()
                    self.cat_fill_[col] = str(mode.iloc[0]) if len(mode) else "unknown"
                except Exception:
                    self.cat_fill_[col] = "unknown"

        return self

    def transform(self, X):
        df = X.copy()

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        # нормализация строк: "" -> NaN (часто приходит из API/CSV)
        for col in df.columns:
            if col in getattr(self, "categorical_cols_", []):
                s = df[col]
                if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
                    s = s.astype(str).str.strip()
                    s_lower = s.str.lower()
                    mask_na = s_lower.isin(["", "nan", "none", "null"])
                    s = s.mask(mask_na, np.nan)
                    df[col] = s

        # применяем статистики с трейна
        for col in df.columns:
            if col in self.numeric_cols_:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(self.num_fill_[col])
            elif col in self.categorical_cols_:
                df[col] = df[col].fillna(self.cat_fill_.get(col, "unknown"))

        return df