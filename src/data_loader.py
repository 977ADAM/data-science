# src/data_loader.py

import pandas as pd
from pathlib import Path
from src.config import RAW_DATA

def load_data(path: str | Path = RAW_DATA) -> pd.DataFrame:
    """
    Загружает данные из CSV.
    path можно переопределять (например, для тестов/пайплайнов).
    """
    return pd.read_csv(path)
