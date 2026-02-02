# src/preprocessing.py

import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # таргет может отсутствовать на инференсе
    if "Churn" in df.columns:
        s = df["Churn"]
        # приводим к устойчивому виду: strip + lower
        s = s.astype(str).str.strip().str.lower()
        df["Churn"] = s.isin(["yes", "y", "1", "true", "t"]).astype(int)
    return df
